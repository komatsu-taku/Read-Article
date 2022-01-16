"""
(1) 作って理解するTransformer 
    Transformerを一つ一つ実装してみた記事 : 今までで読んだ中でもわかりやすかった
    (a) 記事
        https://qiita.com/halhorn/items/c91497522be27bde17ce
    
    (b) Attention概要
        query(input), key・value(memory)とによるattention
        Attention : queryによってmemoryかあ必要な情報を取ってくること。

        query + scale\
                      matmul -> logit -> attention_weight \
              key    /                 ↑
                                    softmax                attention_dense
                                            value         /

    (c) Attentionの種類
        ・Self-Attention
            : query(input)とmemory(key,vlaue)に同じTensorを用いる
            : TransformerのEncoder, Decoder, 文章分類などで使われる
        ・SourceTarget-Attention
            : query(input)とmemory(key,value)は別のTensorを用いる
            : TransformerのDecoder部分で使われる
    
    (d) part
        ・ Scaled Dot-production : softmaxで勾配が0にならないように変換する

        ・ Mask : Padの無視 or Decoderで未来の情報を参照できないようにする
        --> Mask部分はTrue, そうでない部分はFalseの行列を作る

        ・ Multi-head Attention : attentionをheadごとに分割してAttentionを行う

        ・ Hopping : Multi-head Attentionを何度も繰り返し適用する
        --> RNNとは異なり、各Attentionは独立した重み。 : 固定長
        --> 可変長にしたもの : Universary Transformer

        ・ Position-wise FeedGorward Network
        --> 各HoppingのAttentionの後にFNNを挟む
        --> このFNNは全時刻で同じ重みを用いる
        --> FFNは二層で、1層目はhidden_dim*4+Relu, 二層目はhidden_size+Linear

        ・ LayerNormalizeation
        --> Batch Normalization, Layer Norma;ization, Instance Normalizationについて↓
        --> https://qiita.com/amateur2020/items/f2c829677d9764af0b50

        ・ ResidualNormalizationWrapper : 本家論文では'PrePostProcessingWrapper'
        --> LayerNorm Dropout, Residual Connectionなどの正則化をレイヤに施すWrapper

        ・positional Encoding : 文章内のトークンの位置を決めるもの
        --> Positional Encodingを各トークンをEmbeddingしたものに足し合わせる
        --> PE(pos, 2i) = sin(pos / 10000^2i/d_model)
        --> PE(pos, 2i+1) = cos(pos, 10000^2i/d_model)
        --> pos ; 時刻, 2i,2i+1 : Embeddingの何番目の次元か
        --> 相対的な位置PE(pos+k)をPE(pos)の線形関数で表現可能

        ・ Token Embedding
        --> 各トークンはintになっているが、これをEmbedding Vectorに変換する
        --> 本家の論文では、最後にembeddingの隠れ層の次元(hidden_dim)に応じてスケールしている

    (e) Transformer : Encoder と Decoder とに大別できる
        ・ Encoder : 入力トークン列をエンコードする
        --> Self-Attentionによって入力をエンコードする
        --> input -> Token-Embeding -> Add-Positional-Embedding 
                        -> N * {Wrapper(Self-Attention) + Wrapper(FFN)}
        
        ・ Decoder
        --> 入力にSelf-Attentionをかけてから、SourceTarget-AttentionでEncodeの出力を読み込み
        --> これによって、時刻0~tを入力として受けとり、時刻1~t+1を出力する




    (e) 番外編
        ・ 名前空間に関して : C++由来の変数スコープ↓
        --> https://qiita.com/TomokIshii/items/ffe999b3e1a506c396c8
        ・ Python での型宣言
        --> https://qiita.com/papi_tokei/items/2a309d313bc6fc5661c3



"""
