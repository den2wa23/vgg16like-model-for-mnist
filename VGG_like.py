import numpy as np
from collections import OrderedDict
from layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss, BatchNormalization

class VGG_like:
    """
    VGG16モデルの実装例
    conv_param: dict
        'filter_size':3, 'pad':1, 'stride':1 等
    pool_param: dict
        'pool_size':2, 'pad':0, 'stride':2 等
    input_dim: tuple
        (チャンネル数, 高さ, 幅)
    output_size: int
        出力クラス数(例えば ImageNet は1000)
    weight_init_std: float
        重み初期化時の標準偏差
    """
    def __init__(self, 
                 input_dim=(3,224,224),
                 conv_param={'filter_size':3,'pad':1,'stride':1},
                 pool_param={'pool_size':2,'pad':0,'stride':2},
                 output_size=1000,
                 weight_init_std=0.01):
        
        # VGG16のフィルタ構成
        # VGG16は以下のConv層数:
        # Block1:64,64
        # Block2:128,128
        # Block3:256,256,256
        # Block4:512,512,512
        # Block5:512,512,512
        # 合計13層のConv
        # 層数を減らして似ているモデルを再構成
        filter_config = [
            [32,32],
            [64,64],
            [128,128],
            [256]
        ]
        
        self.params = {}
        self.layers = OrderedDict()
        
        current_C, current_H, current_W = input_dim
        std = weight_init_std
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        layer_idx = 1
        
        # Conv-BN-ReLUを繰り返しながら、ブロックごとにMaxPoolを挿入
        # conv層ごとにパラメータW,b,gamma,betaを作成
        for block_idx, block_filters in enumerate(filter_config, start=1):
            for f_num in block_filters:
                # Conv層出力サイズ
                # 出力サイズは(H_out = (H_in + 2*pad - filter_size)/stride + 1)
                conv_output_size = (current_H + 2*filter_pad - filter_size)//filter_stride + 1
                
                # パラメータ初期化
                W_name = 'W' + str(layer_idx)
                b_name = 'b' + str(layer_idx)
                
                self.params[W_name] = std * np.random.randn(f_num, current_C, filter_size, filter_size)
                self.params[b_name] = np.zeros(f_num)
                
                
                self.layers['Conv' + str(layer_idx)] = Convolution(self.params[W_name], self.params[b_name],stride=filter_stride, pad=filter_pad)
                self.layers['ReLU' + str(layer_idx)] = ReLU()
                
                # Update dimension info
                current_C = f_num
                current_H = conv_output_size
                current_W = conv_output_size
                
                layer_idx += 1
            
            # ブロックの最後に正規化
             # ブロックの最後に追加のバッチ正規化
            gamma_name = f'gamma_block{block_idx}'
            beta_name = f'beta_block{block_idx}'
            self.params[gamma_name] = np.ones(current_C)
            self.params[beta_name] = np.zeros(current_C)
            self.layers[f'BatchNorm_block{block_idx}'] = BatchNormalization(self.params[gamma_name], self.params[beta_name])
            # ブロックの最後にプーリング
            self.layers['Pool' + str(block_idx)] = MaxPooling(pool_h=pool_size, pool_w=pool_size,
                                                              stride=pool_stride, pad=pool_pad)
            # PoolingによってH,Wが半分になる
            current_H = (current_H + 2*pool_pad - pool_size)//pool_stride + 1
            current_W = (current_W + 2*pool_pad - pool_size)//pool_stride + 1
        
        # 全結合層(Flatten->Affine->BN->ReLU -> Affine->BN->ReLU -> Affine)
        # 最終的には (7x7x512) = 25088 個の入力ユニットが全結合層へ
        pool_output_pixel = current_C * current_H * current_W
        
        # 全結合1
        self.params['W_fc1'] = std * np.random.randn(pool_output_pixel, 512)
        self.params['b_fc1'] = np.zeros(512)
 
        

        
        # 全結合3 (最後の分類層)
        self.params['W_fc3'] = std * np.random.randn(512, output_size)
        self.params['b_fc3'] = np.zeros(output_size)
        
        # 全結合層レイヤー定義
        self.layers['Affine_fc1'] = Affine(self.params['W_fc1'], self.params['b_fc1'])
        self.layers['ReLU_fc1'] = ReLU()
        
        
        self.layers['Affine_fc3'] = Affine(self.params['W_fc3'], self.params['b_fc3'])
        
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x
    
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=50):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        for i in range(int(x.shape[0]/batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y==tt)
        return acc / x.shape[0]
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        # Conv層(13層分)
        conv_count = 7
        for i in range(1, conv_count+1):
            grads['W'+str(i)] = self.layers['Conv'+str(i)].dW
            grads['b'+str(i)] = self.layers['Conv'+str(i)].db

        
        for block_idx in range(1, 4 + 1):
            if f'BatchNorm_block{block_idx}' in self.layers:
                grads[f'gamma_block{block_idx}'] = self.layers[f'BatchNorm_block{block_idx}'].dgamma
                grads[f'beta_block{block_idx}'] = self.layers[f'BatchNorm_block{block_idx}'].dbeta

        # 全結合層
        grads['W_fc1'] = self.layers['Affine_fc1'].dW
        grads['b_fc1'] = self.layers['Affine_fc1'].db
        
        
        grads['W_fc3'] = self.layers['Affine_fc3'].dW
        grads['b_fc3'] = self.layers['Affine_fc3'].db
        
        return grads