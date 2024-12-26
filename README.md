# vgg16like-model-for-mnist
numpyのみを用いてCNNの分類タスクを行った.VGG16と呼ばれるモデルを参考に作成した.データ拡張にKerasを用いた.

## Requirement
- Python 3.9.13
- Numpy 1.23.5
- Sklearn 1.1.3
- Keras 2.11.0

## model
バッチ正規化をConvolution層のブロック間に配置して最終層に全結合層を配置.

## files
- Activation_func.py softmax関数 
- im2col.py im2colとcol2im関数
- layers.py ReLU,Affine層,Softmax層,バッチ正規化層,畳み込み層,MaxPooling層
- loss_func.py クロスエントロピー誤差
- optimizer.py Adam
- VGG_like.py VGG16の軽量化モデル
- train.ipynb 学習用のノートブック
- predict.ipynb 予測用のノートブック
## result
testデータの精度が0.9903.



