## IA do zero: Gradient Descent — otimizando seu neurônio em rust

<p align="center">
  <img src="https://raw.githubusercontent.com/z4nder/rs-gradient-descent-neuron/main/assets/banner.png" alt="banner do projeto" width="1000" />
</p>

Você já usou o GPT, mas sabe o que existe dentro dele? Neste post vamos ensinar o neurônio a melhorar suas previsões de forma inteligente usando **gradient descent**, o algoritmo que está por trás de praticamente todo o aprendizado de máquina moderno.

### Conteúdo

- 1 [Prólogo](#1)
- 2 [O problema com o ±0.01](#2)
- 3 [Medindo o erro geral — Loss (MSE)](#3)
- 4 [A parábola — visualizando o loss](#4)
- 5 [O gradiente — a inclinação como bússola](#5)
- 6 [Gradient Descent — descendo a parábola](#6)
- 7 [Implementando o Gradient Descent](#7)
- 8 [Resultado — ±0.01 vs Gradient Descent](#8)
- 9 [Conclusão](#9)

---

### 1. Prólogo <a name="1"></a>

Retomando o [post anterior](https://dev.to/z4nder/ia-do-zero-implementar-seu-primeiro-neuronio-em-rust-jc9), implementamos um neurônio capaz de aprender a prever a distância de um projétil com base na energia do disparo. O treinamento e os resultados foram razoáveis, mas o algoritmo de ajuste utilizado era propositalmente simples para fins educacionais.

Agora vamos entender e implementar o tão temido **Gradient Descent**.

Confesso que, durante meu aprendizado desse tópico, me assustei um pouco com as notações matemáticas, uso de derivadas e gráficos em três dimensões para dar um pequeno passo além do que parecia tão simples no último post. Mas, depois de estudar um pouco mais, acredito que consegui entender um pouco melhor o assunto então vou tentar compartilhar esse entendimento aqui.

---

### 2. O problema com o ±0.01 <a name="2"></a>

O treino anterior ajustava `w` e `b` com um passo fixo:

```rust
if error > 0.0 {
    self.weight -= 0.01;
    self.bias   -= 0.01;
} else if error < 0.0 {
    self.weight += 0.01;
    self.bias   += 0.01;
}
```

O problema aparece na prática:

```
erro = 500  → ajusta 0.01
erro = 0.5  → ajusta 0.01
```

Quando o neurônio está muito errado, deveria dar passos grandes e quando está quase certo, passos pequenos para não passar do ponto.

O que precisamos é de um ajuste **proporcional ao erro**. É aí que entra o **loss** e depois o **gradient descent**.

---

### 3. Medindo o erro geral — Loss (MSE) <a name="3"></a>

Já calculamos o erro de um item no dataset de forma simples no post anterior

```
erro = previsto - real
```

Mas precisamos entender o quão errado o neurônio está no conjunto inteiro de dados, e não apenas em um exemplo específico pois se ficarmos ajustando as variaveis somente com base no erro de um par do dataset acabamos prejudicando o resultado em outros então precisamos encontrar uma forma de calcular o erro médio em cima do dataset que aqui chamaremos de **`loss`**.

Para isso precisamos da média dos erros

```
loss = (erro1 + erro2 + erro3) / n
```

O problema é que erros se cancelam:

```
ponto A: erro = +30  (previu demais)
ponto B: erro = -30  (previu de menos)

média = 0  ← parece perfeito, mas está errado nos dois casos
```

A solução é o **MSE — Mean Squared Error** (Erro Quadrático Médio):

```
loss = (1/n) × Σ (previsto - real)²
```

Elevar ao quadrado tem dois efeitos:

1. **Remove o sinal:** erros positivos e negativos não se cancelam mais
2. **Penaliza erros grandes:** um erro de 10 vira 100, um erro de 2 vira 4

```
erro = 2  → contribuição = 4
erro = 10 → contribuição = 100
```

Um erro cinco vezes maior gera uma penalização vinte e cinco vezes maior, o loss deve gritar quando o neurônio está muito errado.

```rust
pub fn loss(dataset: &[(f64, f64)], neuron: &Neuron) -> f64 {
    let n = dataset.len() as f64;
    let sum: f64 = dataset
        .iter()
        .map(|(x, actual)| {
            let error = neuron.predict(*x) - actual;
            error * error
        })
        .sum();
    sum / n
}
```

> **MSE** é a fórmula. **Loss** é o conceito. A partir daqui vamos usar o termo **loss** para falar do erro geral do neurônio.

---

### 4. A parábola — visualizando o loss <a name="4"></a>

Agora que temos o loss, podemos calcular o loss para cada valor possível de `w` e plotar o resultado. Para isso, fixamos `b=18` como estimativa inicial, isolando o efeito de `w` no gráfico.

<img src="https://raw.githubusercontent.com/z4nder/rs-gradient-descent-neuron/main/assets/02_parabola.png" alt="parábola do loss variando w com b fixo em 18, mostrando o mínimo em w=0.92" width="700" />

Podemos observar que o gráfico tem formato de **parábola** caindo até um mínimo e sobe novamente, o ponto onde **`w=0.92`** é o valor ideal para `b=18`, o **mínimo** do loss nesse dataset.

O objetivo do treino é encontrar os valores de **`w`** e **`b`** que reduzem ao máximo o **`loss`**.

Na prática, o loss é uma **superfície 3D** em função de `(w, b)` se tornando algo parecido com uma tigela onde ainda assim nosso objetivo é atingir o ponto mais fundo e com o gradient descent desce essa superfície simultaneamente nos dois eixos.

<img src="https://raw.githubusercontent.com/z4nder/rs-gradient-descent-neuron/main/assets/02_surface.png" alt="superfície 3D do loss em função de w e b, com destaque no ponto mínimo" width="700" />

---

### 5. O gradiente — a inclinação como bússola <a name="5"></a>

Você está em algum ponto da parábola e quer chegar no fundo, o **gradiente** é a inclinação da curva naquele ponto e ele te diz duas coisas:

- **Direção:** se a curva está subindo, você vai na direção oposta
- **Magnitude:** curva íngreme (longe do fundo) → passo grande. Quase plana (perto do fundo) → passo pequeno

```
w=0   → inclinação íngreme → passo grande
w=0.8 → inclinação suave  → passo pequeno
w=0.9 → fundo da parábola → gradiente = 0, para
```

É exatamente o que faltava no `±0.01`: o passo **proporcional à inclinação**, não fixo.

Os gradientes do loss em relação a `w` e `b` são:

```
∂loss/∂w = (2 / dataset_size) * Σ(erro * x)
∂loss/∂b = (2 / dataset_size) * Σ(erro)
```

1. **`w`** multiplica por `x` porque `w` está ligado a `x` na equação do neurônio (`y = wx + b`).
2. **`b`** acumula apenas o erro, pois é um deslocamento constante não depende de `x`.

Não precisa se assustar com essas fórmulas pois não vamos derivá-las agora.

Por enquanto, basta entender que elas medem a inclinação do erro em relação a `w` e `b`, indicando para qual direção devemos mover os parâmetros e o quão forte deve ser essa atualização.

Na implementação, veremos que calcular os gradientes é bem mais simples do que a notação matemática sugere.

---

### 6. Gradient Descent — descendo a parábola <a name="6"></a>

Agora que entendemos que precisamos alterar `w` e `b` em direção ao mínimo podemos aplicar essa fórmula

```
erro = previsto - real
dataset_size = dataset.len

∂L/∂w = (2 / dataset_size) * Σ(erro * x)
∂L/∂b = (2 / dataset_size) * Σ(erro)

Gradient Descent
w = w - lr * ∂loss/∂w
b = b - lr * ∂loss/∂b
```

Assim nasce o **Gradient Descent**, para aplicar vamos primeiro definir um valor de **`lr`**=0.0001, que é o tamanho do passo que vamos dar.

```
lr muito grande → passa do fundo, fica oscilando
lr muito pequeno → chega lá, mas demora muito
lr ideal → desce suave até o mínimo
```

Para aplicar vamos definir alguns valores iniciais

```
w  = 10.0
b  = 10.0
lr = 0.0001
dataset_size = 4
```

Também vamos definir um dataset

```
(1,  60)
(2,  80)
(10, 60)
(4,  70)
```

Com `w=10, b=10` o neurônio calcula `previsto = 10x + 10`. Agora acumulamos `Σ(erro)` e `Σ(erro * x)` ponto a ponto:

| posição |   x | real | previsto | erro = previsto - real | erro × x |
| ------- | --: | ---: | -------: | ---------------------: | -------: |
| 1       |   1 |   60 |       20 |                    -40 |      -40 |
| 2       |   2 |   80 |       30 |                    -50 |     -100 |
| 3       |  10 |   60 |      110 |                     50 |      500 |
| 4       |   4 |   70 |       50 |                    -20 |      -80 |

Calculando os somatórios `Σ(erro)` e `Σ(erro * x)`

```
Σ(erro * x) = -40 + -100 + 500 + -80 = 280
Σ(erro)     = -40 +  -50 +  50 + -20 = -60
```

Calculando as derivadas `∂L/∂w` e `∂L/∂b`

```
∂L/∂w = (2 / 4) * 280 = 140
∂L/∂b = (2 / 4) * -60 = -30
```

Calculando **Gradient Descent**

```
weight = 10.0 - 0.0001 * 140    → 9.986
bias   = 10.0 - 0.0001 * (-30)  → 10.003
```

Chegamos em um ajuste de `w=10` para `w=9.986` e `b=10` para `b=10.003`. Com `∂L/∂w = +140` o gradiente é positivo, então diminuímos `w`, o neurônio estava prevendo demais em `x=10`. Com `∂L/∂b = -30` o gradiente é negativo, então aumentamos `b`, a maioria dos pontos estava sendo subestimada.

### 7. Implementando o Gradient Descent <a name="7"></a>

Agora que calculamos manualmente os valores chegou a parte mais fácil que é implementar nosso algoritmo.

**1. Inicializa `Σ(erro * x)` e `Σ(erro)`**

```rust
let mut error_x_sum = 0.0;  // Σ(erro * x)
let mut error_sum   = 0.0;  // Σ(erro)
```

**2. Acumula os erros do dataset**

Para cada ponto calculamos o erro e acumulamos os dois somatórios.

```rust
for (x, actual) in dataset {
    let error = self.predict(*x) - actual;

    error_x_sum += error * x;
    error_sum   += error;
}
```

**3. Calcula as derivadas `∂L/∂w` e `∂L/∂b`**

Transformamos os somatórios nos gradientes finais dividindo pelo tamanho do dataset.

```rust
let grad_w = (2.0 / dataset_size) * error_x_sum;  // ∂L/∂w
let grad_b = (2.0 / dataset_size) * error_sum;     // ∂L/∂b
```

**4. Aplica o Gradient Descent**

Andamos na direção oposta ao gradiente (por isso o `-`), com passo controlado pelo **`lr`**.

- `grad_w` positivo → diminuímos `w`
- `grad_w` negativo → aumentamos `w`
- `grad_w` próximo de zero → estamos perto do mínimo

```rust
self.weight -= lr * grad_w;
self.bias   -= lr * grad_b;
```

**5. Implementação completa**

```rust
pub fn train(&mut self, dataset: &[(f64, f64)], lr: f64, epochs: usize) {
    let dataset_size = dataset.len() as f64;

    for _epoch in 0..epochs {
        let mut error_x_sum = 0.0;
        let mut error_sum   = 0.0;

        for (x, actual) in dataset {
            let error = self.predict(*x) - actual;
            error_x_sum += error * x;
            error_sum   += error;
        }

        let grad_w = (2.0 / dataset_size) * error_x_sum;
        let grad_b = (2.0 / dataset_size) * error_sum;

        self.weight -= lr * grad_w;
        self.bias   -= lr * grad_b;
    }
}
```

---

### 8. Resultado — ±0.01 vs Gradient Descent <a name="8"></a>

Rodamos os dois algoritmos com as mesmas condições iniciais:

|               | ±0.01       | Gradient Descent          |
| ------------- | ----------- | ------------------------- |
| Epochs        | 1000        | 1000                      |
| Passo         | fixo `0.01` | proporcional ao gradiente |
| Learning rate | —           | `0.0003`                  |
| W inicial     | 5.0         | 5.0                       |
| B inicial     | 5.0         | 5.0                       |

Com isso obtivemos os seguintes resultados

<img src="https://raw.githubusercontent.com/z4nder/rs-gradient-descent-neuron/main/assets/02_comparison.png" alt="comparação dos ajustes de ±0.01 e gradient descent sobre o dataset" width="700" />

O gradient descent não fez a linha tocar em todos os pontos e isso é esperado. Os dados não seguem uma reta perfeita, então nenhuma combinação de `w` e `b` vai passar por todos. Isso é uma limitação do modelo, não do treino.

O que o gradient descent faz é encontrar a **melhor reta possível** dentro dessa limitação e o loss mostra isso claramente caindo de `77` no `±0.01` para `37`.

<img src="https://raw.githubusercontent.com/z4nder/rs-gradient-descent-neuron/main/assets/02_loss_comparison.png" alt="curva de loss ao longo das epochs comparando ±0.01 e gradient descent em escala log" width="700" />

Outro ponto que vale destacar é a forma como o gradient descent realiza os ajustes de `w` e `b`, no gráfico abaixo podemos observar isso

<img src="https://raw.githubusercontent.com/z4nder/rs-gradient-descent-neuron/main/assets/02_path.png" alt="caminho de cada algoritmo na parábola epoch a epoch, mostrando passos fixos vs proporcionais" width="700" />

Cada ponto é uma epoch. Com `±0.01` dá passos iguais o tempo todo, mesmo perto do fundo continua com a mesma força, já o gradient descent desacelera conforme se aproxima e para quando o gradiente chega a zero.

Ainda sim a linha não tocou perfeitamente todos os pontos e para isso seriam necessários mais parâmetros e mais neurônios, uma rede neural. Isso vem nas próximas fases.

> Para a linha tocar em todos os pontos seriam necessários mais parâmetros mais neurônios, uma rede neural. Isso vem nas próximas fases.

---

### 9. Conclusão <a name="9"></a>

Neste post saímos de um ajuste cego e implementamos **gradient descent**, o algoritmo base do aprendizado de máquina moderno.

O que foi aprendido:

- O **loss** (MSE) transforma os erros individuais num único número que representa o desempenho geral do neurônio
- O **gradiente** é a inclinação do loss em relação a cada parâmetro e ele diz a direção e o tamanho do passo
- O gradient descent desce a superfície de loss simultaneamente em `w` e `b` até encontrar o mínimo

O que ainda não resolvemos: com um único neurônio e uma função linear, o melhor que conseguimos é uma reta e dados que não seguem uma reta linear não podem ser modelados assim, independente de quantas **epochs** ou de qual algoritmo de treino.

No próximo post: **redes neurais**, múltiplos neurônios em camadas que juntos conseguem aprender padrões não-lineares.

---

### Referências

- [Código-fonte do projeto](https://github.com/z4nder/rs-gradient-descent-neuro)
- [Neural Network from Scratch — vídeo que inspirou essa série](https://www.youtube.com/watch?v=GkiITbgu0V0&t=477s)
- [Gradient Descent — Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)

---

Se este post fizer sentido pra você, o próximo passo natural é adicionar mais neurônios e introduzir funções de ativação é aí que o aprendizado começa a capturar padrões que uma reta simples não consegue descrever.
