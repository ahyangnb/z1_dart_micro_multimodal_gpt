import 'dart:collection';
import 'dart:io';
import 'dart:math' as math;

// 显微镜级多模态 GPT。
//
// 它不是商用 VLM，也不读取真实照片；它把“图片”压缩成 4x4 黑白像素，
// 再切成 2x2 patch token。这样可以完整看见：
//
// 1. image -> patch tokens
// 2. text -> character tokens
// 3. image tokens + text tokens -> one causal sequence
// 4. self-attention 融合图片和文字
// 5. lm_head 预测下一个文字 token
// 6. 自回归地生成答案

final rng = math.Random(2026);

const nEmbd = 12;
const nHead = 3;
const headDim = nEmbd ~/ nHead;
const hiddenDim = nEmbd * 3;
const maxSeqLen = 24;
const patchSize = 2;
const patchDim = patchSize * patchSize;
const defaultSteps = 200;

late TinyTokenizer tokenizer;
late List<Value> params;

final stateDict = <String, List<List<Value>>>{};

class Value {
  Value(num data, [List<Value>? children, List<double>? localGrads])
      : data = data.toDouble(),
        _children = children ?? const <Value>[],
        _localGrads = localGrads ?? const <double>[];

  double data;
  double grad = 0.0;
  final List<Value> _children;
  final List<double> _localGrads;

  Value operator +(Object other) {
    final o = _asValue(other);
    return Value(data + o.data, <Value>[this, o], <double>[1.0, 1.0]);
  }

  Value operator *(Object other) {
    final o = _asValue(other);
    return Value(data * o.data, <Value>[this, o], <double>[o.data, data]);
  }

  Value operator -() => this * -1.0;

  Value operator -(Object other) => this + (-_asValue(other));

  // pow 是 power 的缩写，表示“幂运算”。x.pow(-1.0) 就是 x 的 -1 次方，
  // 等价于 1 / x，所以这里用“乘以倒数”来实现除法。
  Value operator /(Object other) => this * _asValue(other).pow(-1.0);

  Value pow(double exponent) {
    // data^exponent，同时记录导数：d(x^n)/dx = n * x^(n - 1)，供反向传播使用。
    final out = math.pow(data, exponent).toDouble();
    final localGrad = exponent * math.pow(data, exponent - 1).toDouble();
    return Value(out, <Value>[this], <double>[localGrad]);
  }

  Value log() => Value(math.log(data), <Value>[this], <double>[1.0 / data]);

  Value exp() {
    final out = math.exp(data);
    return Value(out, <Value>[this], <double>[out]);
  }

  Value relu() {
    return Value(
      math.max(0.0, data),
      <Value>[this],
      <double>[data > 0.0 ? 1.0 : 0.0],
    );
  }

  void backward() {
    final topo = <Value>[];
    final visited = HashSet<Value>.identity();

    void buildTopo(Value v) {
      if (visited.add(v)) {
        for (final child in v._children) {
          buildTopo(child);
        }
        topo.add(v);
      }
    }

    buildTopo(this);
    grad = 1.0;

    for (final v in topo.reversed) {
      for (var i = 0; i < v._children.length; i++) {
        v._children[i].grad += v._localGrads[i] * v.grad;
      }
    }
  }
}

Value _asValue(Object other) {
  if (other is Value) return other;
  if (other is num) return Value(other);
  throw ArgumentError.value(other, 'other', 'Expected a Value or num.');
}

class ImageGrid {
  const ImageGrid({
    required this.id,
    required this.label,
    required this.rows,
  });

  final String id;
  final String label;
  final List<String> rows;

  int get height => rows.length;
  int get width => rows.first.length;

  List<Patch> patches() {
    final result = <Patch>[];
    var index = 0;

    for (var y = 0; y < height; y += patchSize) {
      for (var x = 0; x < width; x += patchSize) {
        final pixels = <double>[];
        for (var dy = 0; dy < patchSize; dy++) {
          for (var dx = 0; dx < patchSize; dx++) {
            pixels.add(rows[y + dy][x + dx] == '1' ? 1.0 : 0.0);
          }
        }
        result.add(Patch(index: index, x: x, y: y, pixels: pixels));
        index++;
      }
    }

    return result;
  }

  void printPixels() {
    for (final row in rows) {
      final pretty = row.runes
          .map((rune) => String.fromCharCode(rune) == '1' ? '#' : '.')
          .join(' ');
      print('   $pretty');
    }
  }
}

class Patch {
  const Patch({
    required this.index,
    required this.x,
    required this.y,
    required this.pixels,
  });

  final int index;
  final int x;
  final int y;
  final List<double> pixels;

  double get mean => pixels.reduce((a, b) => a + b) / pixels.length;

  String get bits => pixels.map((v) => v > 0.5 ? '1' : '0').join();
}

class TrainingExample {
  const TrainingExample({
    required this.image,
    required this.prompt,
    required this.answer,
  });

  final ImageGrid image;
  final String prompt;
  final String answer;
}

class TinyTokenizer {
  TinyTokenizer._(this.idToToken)
      : tokenToId = {
          for (var i = 0; i < idToToken.length; i++) idToToken[i]: i,
        };

  final List<String> idToToken;
  final Map<String, int> tokenToId;

  int get sep => tokenToId['<sep>']!;
  int get bos => tokenToId['<bos>']!;
  int get eos => tokenToId['<eos>']!;

  int get vocabSize => idToToken.length;

  static TinyTokenizer fromExamples(List<TrainingExample> examples) {
    final chars = <String>{};
    for (final example in examples) {
      chars.addAll(charsOf(example.prompt));
      chars.addAll(charsOf(example.answer));
    }

    final sortedChars = chars.toList()..sort();
    return TinyTokenizer._(['<sep>', '<bos>', '<eos>', ...sortedChars]);
  }

  List<int> encode(String text) {
    return [
      for (final ch in charsOf(text))
        if (tokenToId.containsKey(ch)) tokenToId[ch]!,
    ];
  }

  String decode(List<int> ids) {
    final out = <String>[];
    for (final id in ids) {
      if (id == sep || id == bos || id == eos) continue;
      out.add(idToToken[id]);
    }
    return out.join();
  }
}

class ForwardResult {
  const ForwardResult({
    required this.logitsByPosition,
    required this.sequenceLabels,
    required this.lastAttention,
  });

  final List<List<Value>> logitsByPosition;
  final List<String> sequenceLabels;
  final List<double> lastAttention;
}

void main(List<String> args) {
  final examples = buildTrainingExamples();
  tokenizer = TinyTokenizer.fromExamples(examples);
  initStateDict(tokenizer.vocabSize);
  params = [
    for (final mat in stateDict.values)
      for (final row in mat)
        for (final p in row) p,
  ];

  final targetImageId = args.isEmpty ? 'sun' : args.first;
  final prompt = args.length <= 1 ? '图里有什么' : args.skip(1).join('');
  final matchedImages = demoImages.where((image) => image.id == targetImageId);
  if (matchedImages.isEmpty) {
    print('未知图片 id: $targetImageId');
    print('可用图片: ${demoImages.map((image) => image.id).join(', ')}');
    return;
  }
  final targetImage = matchedImages.first;

  print('--- 显微镜级多模态 GPT ---');
  print('vocab size: ${tokenizer.vocabSize}');
  print('num params: ${params.length}');
  print('image patches: ${targetImage.patches().length}');
  print('');

  final steps = intFromEnvironment('Z1_DART_MM_GPT_STEPS', defaultSteps);
  train(examples, steps);
  printMicroscopeRun(targetImage, prompt);
}

List<TrainingExample> buildTrainingExamples() {
  const prompts = ['图里有什么', '这是什么', '图片内容'];
  return [
    for (final image in demoImages)
      for (final prompt in prompts)
        TrainingExample(image: image, prompt: prompt, answer: image.label),
  ];
}

int intFromEnvironment(String name, int defaultValue) {
  final raw = Platform.environment[name];
  if (raw == null) return defaultValue;
  final parsed = int.tryParse(raw);
  return parsed == null || parsed < 1 ? defaultValue : parsed;
}

List<String> charsOf(String s) {
  return s.runes.map(String.fromCharCode).toList();
}

double randomGaussian(double mean, double std) {
  var u1 = 0.0;
  while (u1 == 0.0) {
    u1 = rng.nextDouble();
  }
  final u2 = rng.nextDouble();
  final z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  return mean + std * z0;
}

List<List<Value>> matrix(int nout, int nin, [double std = 0.08]) {
  return List.generate(
    nout,
    (_) => List.generate(nin, (_) => Value(randomGaussian(0.0, std))),
  );
}

void initStateDict(int vocabSize) {
  stateDict['wte'] = matrix(vocabSize, nEmbd);
  stateDict['wpe'] = matrix(maxSeqLen, nEmbd);
  stateDict['wme'] = matrix(2, nEmbd);
  stateDict['vision_proj'] = matrix(nEmbd, patchDim);
  stateDict['attn_wq'] = matrix(nEmbd, nEmbd);
  stateDict['attn_wk'] = matrix(nEmbd, nEmbd);
  stateDict['attn_wv'] = matrix(nEmbd, nEmbd);
  stateDict['attn_wo'] = matrix(nEmbd, nEmbd);
  stateDict['mlp_fc1'] = matrix(hiddenDim, nEmbd);
  stateDict['mlp_fc2'] = matrix(nEmbd, hiddenDim);
  stateDict['lm_head'] = matrix(vocabSize, nEmbd);
}

List<Value> linear(List<Value> x, List<List<Value>> w) {
  return [for (final row in w) dot(row, x)];
}

Value dot(List<Value> a, List<Value> b) {
  var out = a[0] * b[0];
  for (var i = 1; i < a.length; i++) {
    out = out + (a[i] * b[i]);
  }
  return out;
}

Value sumValues(Iterable<Value> values) {
  final iterator = values.iterator;
  if (!iterator.moveNext()) return Value(0.0);

  var total = iterator.current;
  while (iterator.moveNext()) {
    total = total + iterator.current;
  }
  return total;
}

List<Value> softmax(List<Value> logits) {
  final maxVal = logits.map((val) => val.data).reduce(math.max);
  final exps = [for (final val in logits) (val - maxVal).exp()];
  final total = sumValues(exps);
  return [for (final e in exps) e / total];
}

List<Value> rmsnorm(List<Value> x) {
  final ms = sumValues([for (final xi in x) xi * xi]) / x.length;
  // (ms + eps)^(-0.5) 等价于 1 / sqrt(ms + eps)，用于把向量按 RMS 缩放。
  final scale = (ms + 1e-5).pow(-0.5);
  return [for (final xi in x) xi * scale];
}

List<Value> addVectors(List<Value> a, List<Value> b) {
  return [for (var i = 0; i < a.length; i++) a[i] + b[i]];
}

List<Value> embedImagePatch(Patch patch, int position) {
  final projected = linear(
    [for (final pixel in patch.pixels) Value(pixel)],
    stateDict['vision_proj']!,
  );
  final withPosition = addVectors(projected, stateDict['wpe']![position]);
  return addVectors(withPosition, stateDict['wme']![0]);
}

List<Value> embedTextToken(int tokenId, int position) {
  final withPosition =
      addVectors(stateDict['wte']![tokenId], stateDict['wpe']![position]);
  return addVectors(withPosition, stateDict['wme']![1]);
}

ForwardResult forward(
  ImageGrid image,
  List<int> textTokenIds, {
  bool captureAttention = false,
}) {
  final patches = image.patches();
  final sequence = <List<Value>>[];
  final labels = <String>[];

  for (final patch in patches) {
    final position = sequence.length;
    sequence.add(embedImagePatch(patch, position));
    labels.add('img${patch.index}');
  }

  for (final tokenId in textTokenIds) {
    final position = sequence.length;
    sequence.add(embedTextToken(tokenId, position));
    labels.add(tokenizer.idToToken[tokenId]);
  }

  if (sequence.length > maxSeqLen) {
    throw StateError('Sequence length ${sequence.length} exceeds $maxSeqLen.');
  }

  var x = sequence;
  final attnInput = [for (final token in x) rmsnorm(token)];
  final qs = [
    for (final token in attnInput) linear(token, stateDict['attn_wq']!)
  ];
  final ks = [
    for (final token in attnInput) linear(token, stateDict['attn_wk']!)
  ];
  final vs = [
    for (final token in attnInput) linear(token, stateDict['attn_wv']!)
  ];

  final attnOut = <List<Value>>[];
  final lastHeadWeights = <List<double>>[];

  for (var i = 0; i < x.length; i++) {
    final heads = <Value>[];
    for (var h = 0; h < nHead; h++) {
      final start = h * headDim;
      final qH = qs[i].sublist(start, start + headDim);
      final logits = [
        for (var j = 0; j <= i; j++)
          dot(qH, ks[j].sublist(start, start + headDim)) / math.sqrt(headDim),
      ];
      final weights = softmax(logits);

      if (captureAttention && i == x.length - 1) {
        lastHeadWeights.add([for (final weight in weights) weight.data]);
      }

      for (var d = 0; d < headDim; d++) {
        heads.add(
          sumValues([
            for (var j = 0; j <= i; j++) weights[j] * vs[j][start + d],
          ]),
        );
      }
    }
    attnOut.add(linear(heads, stateDict['attn_wo']!));
  }

  x = [
    for (var i = 0; i < x.length; i++) addVectors(x[i], attnOut[i]),
  ];

  final mlpInput = [for (final token in x) rmsnorm(token)];
  final mlpOut = <List<Value>>[];
  for (final token in mlpInput) {
    var y = linear(token, stateDict['mlp_fc1']!);
    y = [for (final yi in y) yi.relu()];
    y = linear(y, stateDict['mlp_fc2']!);
    mlpOut.add(y);
  }

  x = [
    for (var i = 0; i < x.length; i++) addVectors(x[i], mlpOut[i]),
  ];

  final logitsByPosition = [
    for (final token in x) linear(rmsnorm(token), stateDict['lm_head']!),
  ];

  final lastAttention = <double>[];
  if (captureAttention && lastHeadWeights.isNotEmpty) {
    for (var i = 0; i < lastHeadWeights.first.length; i++) {
      final total = lastHeadWeights.fold(0.0, (sum, head) => sum + head[i]);
      lastAttention.add(total / lastHeadWeights.length);
    }
  }

  return ForwardResult(
    logitsByPosition: logitsByPosition,
    sequenceLabels: labels,
    lastAttention: lastAttention,
  );
}

List<int> trainingTextInput(TrainingExample example) {
  return [
    ...tokenizer.encode(example.prompt),
    tokenizer.sep,
    tokenizer.bos,
    ...tokenizer.encode(example.answer),
  ];
}

List<int> answerTargets(TrainingExample example) {
  return [...tokenizer.encode(example.answer), tokenizer.eos];
}

void train(List<TrainingExample> examples, int steps) {
  const learningRate = 0.018;
  const beta1 = 0.85;
  const beta2 = 0.98;
  const epsAdam = 1e-8;

  final m = List.filled(params.length, 0.0);
  final v = List.filled(params.length, 0.0);

  for (var step = 0; step < steps; step++) {
    final example = examples[rng.nextInt(examples.length)];
    final textIds = trainingTextInput(example);
    final targets = answerTargets(example);
    final answerStart = tokenizer.encode(example.prompt).length + 1;
    final result = forward(example.image, textIds);

    final losses = <Value>[];
    for (var i = 0; i < targets.length; i++) {
      final globalPos = example.image.patches().length + answerStart + i;
      final probs = softmax(result.logitsByPosition[globalPos]);
      losses.add(-probs[targets[i]].log());
    }

    final loss = sumValues(losses) * (1.0 / losses.length);
    loss.backward();

    final lrT = learningRate * (1.0 - 0.7 * step / steps);
    for (var i = 0; i < params.length; i++) {
      final p = params[i];
      m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1.0 - beta2) * p.grad * p.grad;

      // beta^(step + 1) 表示衰减系数累计乘了多少轮，用于 Adam 的偏差修正。
      final mHat = m[i] / (1.0 - math.pow(beta1, step + 1));
      final vHat = v[i] / (1.0 - math.pow(beta2, step + 1));
      p.data -= lrT * mHat / (math.sqrt(vHat) + epsAdam);
      p.grad = 0.0;
    }

    final shouldPrint = step == 0 || (step + 1) % math.max(1, steps ~/ 5) == 0;
    if (shouldPrint) {
      print(
        'train step ${(step + 1).toString().padLeft(4)} / '
        '${steps.toString().padLeft(4)} | loss ${loss.data.toStringAsFixed(4)}',
      );
    }
  }
  print('');
}

List<int> generate(ImageGrid image, String prompt, {int maxNewTokens = 6}) {
  final textIds = [
    ...tokenizer.encode(prompt),
    tokenizer.sep,
    tokenizer.bos,
  ];
  final generated = <int>[];

  for (var step = 0; step < maxNewTokens; step++) {
    final result = forward(image, textIds);
    final probs = softmax(result.logitsByPosition.last);
    final nextId = argmax([for (final p in probs) p.data]);
    if (nextId == tokenizer.eos) break;
    textIds.add(nextId);
    generated.add(nextId);
  }

  return generated;
}

int argmax(List<double> values) {
  var bestIndex = 0;
  var bestValue = values.first;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestIndex = i;
      bestValue = values[i];
    }
  }
  return bestIndex;
}

void printMicroscopeRun(ImageGrid image, String prompt) {
  print('1. 原始图片: ${image.id}，训练标签: ${image.label}');
  image.printPixels();
  print('');

  print('2. image -> 2x2 patch tokens');
  for (final patch in image.patches()) {
    print(
      '   img${patch.index}: xy=(${patch.x},${patch.y}) '
      'pixels=${patch.bits} mean=${patch.mean.toStringAsFixed(2)}',
    );
  }
  print('');

  final promptIds = tokenizer.encode(prompt);
  print('3. text -> character tokens');
  print('   prompt: $prompt');
  for (var i = 0; i < promptIds.length; i++) {
    print(
        '   text$i: "${tokenizer.idToToken[promptIds[i]]}" -> ${promptIds[i]}');
  }
  print('');

  final prefixIds = [...promptIds, tokenizer.sep, tokenizer.bos];
  final traced = forward(image, prefixIds, captureAttention: true);
  print('4. 多模态 causal sequence');
  for (var i = 0; i < traced.sequenceLabels.length; i++) {
    print('   pos ${i.toString().padLeft(2)}: ${traced.sequenceLabels[i]}');
  }
  print('');

  print('5. 最后一个 token (<bos>) 对前文的 attention');
  final pairs = [
    for (var i = 0; i < traced.lastAttention.length; i++)
      MapEntry(traced.sequenceLabels[i], traced.lastAttention[i]),
  ]..sort((a, b) => b.value.compareTo(a.value));
  for (final pair in pairs.take(8)) {
    print('   ${pair.key.padRight(6)} weight=${pair.value.toStringAsFixed(3)}');
  }
  print('');

  print('6. lm_head softmax: 下一个字的候选概率');
  final probs = softmax(traced.logitsByPosition.last);
  final ranked = [
    for (var i = 0; i < probs.length; i++) MapEntry(i, probs[i].data),
  ]..sort((a, b) => b.value.compareTo(a.value));
  for (final item in ranked.take(8)) {
    print(
      '   "${tokenizer.idToToken[item.key]}" '
      'p=${item.value.toStringAsFixed(3)}',
    );
  }
  print('');

  final outputIds = generate(image, prompt);
  print('7. 自回归生成');
  print('   answer token ids: ${outputIds.join(', ')}');
  print('   answer text: ${tokenizer.decode(outputIds)}');
}

const demoImages = [
  ImageGrid(
    id: 'sun',
    label: '太阳',
    rows: [
      '0110',
      '1111',
      '1111',
      '0110',
    ],
  ),
  ImageGrid(
    id: 'tree',
    label: '树',
    rows: [
      '0110',
      '1111',
      '0110',
      '0100',
    ],
  ),
  ImageGrid(
    id: 'house',
    label: '房子',
    rows: [
      '0110',
      '1111',
      '1001',
      '1111',
    ],
  ),
  ImageGrid(
    id: 'cross',
    label: '十字',
    rows: [
      '0100',
      '1110',
      '0100',
      '0100',
    ],
  ),
];
