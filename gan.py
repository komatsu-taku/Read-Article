"""
(1) 今さら聞けないGAN (!) 基本構造の理解
    コードレベルで理解するように説明した記事
    https://qiita.com/triwave33/items/1890ccc71fab6cbca87e


(2) はじめてのGAN
    詳細に解説された記事
     ://elix-tech.github.io/ja/2017/02/06/gan.html

    (A) 基礎理論
        ・生成モデル
            訓練データを学習し、似たような新しいデータを生成するモデルの総称。
            一般的に、モデルが持つパラメータに対して訓練データの量が圧倒的に多いため、
            モデルは重要な特徴量を学習することが求められる。
            ex) GAN, VAE....
        
        ・GANの仕組み
            Genterator と Discriminatorの二つのネットワークからなる。
            Generator : 訓練データと似たようなデータを生成する
            Discriminator : 訓練データからきたものか生成モデルからきたものかを見分ける
            --> 最終的に、discriminatorの正答率が50%となるのが理想

            GAN の構造
            z(noise) -> G \
                            D
                        x /
            
            損失関数
            minmaxV(D, G) = E_{x~pdata}(x)[logD(x)] + E_{z~pz(z)}(z)[1-D(G(z))]
             G  D
            --> D(x) : xが訓練データである確率を表す
            --> DはD(x)を最大化 == 正しく分別できるようにする
            --> Gはlog(1-D(G(z)))を最小化する == うまく騙せるようにする
    
    (B) DCGAN (Deep Convolutional GAN)
        --> CNNを取り込んだモデル
        
        ・LAPGAN
            CNNを使ったGANで最初に高解像度画像の生成に成功したモデル
            but 何段階にも分けて画像を生成する必要があった。
        
        ・DCGAN
            一発で高解像度画像を生成することに成功したモデル

            (a) プーリングをやめる
                CNNでは、MaxPoolingを用いてダウンサンプリングを行うのが主流。
                discriminatorでは、stride=2の畳み込みに置き換えている。
                generatorでは、fractionally-strided convolutionを使ってupsamplingを行う
                (deconvolutionということもあるが、他の手法を指すため、厳密には異なる)
            
            (b) 全結合層をなくす
                CNNでは、最後の方の層に全結合層を用いることが主流。
                discriminatorでは、全結合層をglobal average poolingに置き換えている。
                # Gloobal Average Poolingに関してQiita記事↓
                # https://qiita.com/mine820/items/1e49bca6d215ce88594a
                最後の各チャンネル(面)の画素平均を求め、それをまとめる手法
                --> パラメータが少なくなるため、過学習が抑制される。一方で、収束が遅くなる
            
            (c) BatchNormlizationを用いる。
                各層でのデータ分布を正規化することで、学習の高速化や、パラメータの初期化をそれほど気にしなくて
                済むようになる。また、過学習を防ぐ効果がある。
                but Generatorの出力層と、Discriminatorの入力層には適用しない。
            
            (d) Leaky ReLUを用いる。
                Generatorは出力層の活性化関数としtanh, それ以外の層ではReLUを用いている。
                Discriminatorでは、すべての層でLeaky ReLUを用いる。
                
                ReLU : f(x) = max(0, x)
                --> x<=0 の領域で勾配が0になるため、学習がストップしてしまう。

                Leaky ReLU : f(x) = max(ax, x) (a==0.2がよく使われる)
                --> x<=0の領域でも学習が進む

                PReLU : f(x) = max(ax, x) + aも含めて学習
                --> 過学習を起こすリスクがある。

    (C) 特徴
        ・ ベクトル演算
            GANにおける入力zベクトルを使ってベクトル演算を行うことが可能
            ex) サングラスをかけた男 - 男 + 女 = サングラスをかけた女
    
    (D) 実装
        --> (1) 記事参照

        ・discriminatorのBatchNormalication関して
            BatchNormalizationを用いてしまうと、discriminatorが強くなりすぎて、うまくいかない
            というリスクがありうる。
    
    (E) その他の派生モデル
        (a) LAPGAN -- 2015
        (b) SRGAN -- 2016
        (c) pix2pix -- 2016
        (d) StackGAN -- 2016
        (e) SimGAN -- 2016

"""
