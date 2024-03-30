# Gpr_details
このプロジェクトは、 log-marginal likelihoodでガウス過程回帰におけるハイパーパラメータのチューニングを説明するためのものである。 

# ガウス分布

一次元ガウス分布の確率密度関数 $p(x)$  

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/5886b7dd-ce40-42c9-83ab-4f717af00cd5)


互いに独立であるn次元ガウス分布と仮定するの場合  

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/39725785-c8d6-4612-a60a-606e125a85b5)

そして、

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/3faac7f1-4e35-4b03-929a-244e7323baeb)

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/812755b6-279b-4752-a846-1a2751106fae)

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/5f09608f-c2cc-4279-ab4d-1df5e2185c0f)

上記の表現を式2に代入すると、次のようになる。

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/096a47ca-21a4-46f7-873c-b65a2269daf2)

$\mathbf{\mu} \in \mathbb{R}^n$は平均値ベクトル, $\mathbf{K} \in \mathbb{R}^{n\times n}$は多次元ガウシアン分布の共分散行列。
簡単に表示すると

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/12c35d45-b564-485a-b12e-b499d5453afd)

各次元の
$x_i$
を関数
$f(x)$
の異なる位置として理解するならば、すべての
$x = [x_1,x_2,...,x_n]$
をサンプリングすると、
$f(x) = [f(x_1), f(x_2),...f(x_n)]$
は次の分布に従う。

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/04f6f90a-d105-4462-b2d7-e9658329cffc)


# カーネル関数
カーネル関数はガウス過程の性質を決定する。 カーネル関数は共分散行列を生成し、ガウス過程における任意の2点間の「距離」を測定する。 異なるカーネル関数は異なる尺度を持ち、ガウス過程の異なる特性を与える。 最もよく使われるカーネル関数の1つはガウシアンカーネル関数で、 放射基底関数 RBFとしても知られ、次のような基本形を持っている。 

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/4a8aa071-b216-47f0-aba7-6813d563f782)

```
def gaussian_kernel(x1, x2, l=1.0, sigma_f=1.0):
    """Easy to understand but inefficient."""
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma_f ** 2 * np.exp(- 0.5 / l ** 2 * dist_matrix)
```

$\sigma^{2}$と
$l$
はカーネルのハイパーパラメータである。

# ガウシアン過程回帰

 - 事前分布 $f(x)$

   ![image](https://github.com/keianxiao/Gpr_details/assets/103640304/abf98285-991c-4b3c-89f1-2db0eb150257)

   観測データ
   $(\mathbf{x^{\star}},\mathbf{y^{\star}})$
   により

    ![image](https://github.com/keianxiao/Gpr_details/assets/103640304/7a42c7c8-2a10-4c0e-a3fd-1c9011edefe1)

   その内

   ![image](https://github.com/keianxiao/Gpr_details/assets/103640304/2bcc02b9-d08a-4fc4-9af5-10e5165a196c)



   ![image](https://github.com/keianxiao/Gpr_details/assets/103640304/c023bab9-66b9-4467-a000-6cb92a4579e7)

ここでは例を挙げて説明する、目的関数は以下のように定義する

```
def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()
```

ハイパーパラメータを適当に設定したガウス過程回帰の結果は、以下のとおりである。

![Gpr_img1](https://github.com/keianxiao/Gpr_details/assets/103640304/2b9f138b-fed8-444f-aba3-5abcf5577889)

半透明のバンドは95%信頼区間を示す。

# ハイパーパラメータ最適化 with Maximum log-marginal likelihood

最適なカーネル関数のハイパーパラメータを選ぶ根拠は、2つのハイパーパラメータでの
$f(x) = y$
出現確率を最大化することである。つまり、信頼区間は最も狭くなる、誤差も最も低めように取られる。式で表すと

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/d4c08acf-8a08-4175-b48e-e5bbafa094ea)

この式の最大値を解く一般的な方法は勾配法であり、通常は共役勾配法SCG(Scaled Conjugate Gradient)と準ニュートン法L-BFGS(Limited-memory Broyden-Fletcher-Goldfarb-Shanno)である。これらのアプローチはいずれも、最大値ではなく極値に収束する確率のある勾配法を用いている。 
```
from scipy.optimize import minimize
class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

```

L-BFGS法を用いて最適化した結果です。

![Gpr_img2](https://github.com/keianxiao/Gpr_details/assets/103640304/607025b2-b25c-4fb3-b2cd-3b356fcefdf4)


一般に、初期点の位置をランダムに選択し、それを何度も繰り返し、すべての繰り返しの最大シナリオを最適化結果とすることによって行われる。私自身の研究では、データの分布が複雑でなかったためか、SCGとL-BFGSの両方で複数回実施しても、結果は基本的に同じであった。SCGに比べ、L-BFGSは収束が少し早い。

![image](https://github.com/keianxiao/Gpr_details/assets/103640304/aaeb1e4a-94bc-424b-9976-610e96226928)

# 原稿での考察

以上の説明によると、ハイパーパラメータの最適化は、勾配法を用いて対数尤度を最大化した後に完了し、原稿の点線は、この手法で最適化したモデルが、誤差について定義した式においての表現である

誤差の定義式

```
L = L_train + a*L_validation
```

今の検討で、もしかしたらFMQAを用いてそのまま対数尤度
$p(\mathbf{y}|\mathbf{x},\sigma, l)$
を最適化する手もありそうです。

