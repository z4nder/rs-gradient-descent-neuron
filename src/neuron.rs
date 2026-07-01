pub struct Config {
    pub epochs: usize,
    pub lr: f64,
    pub w0: f64,
    pub b0: f64,
}

pub struct Neuron {
    pub weight: f64,
    pub bias: f64,
}

impl Neuron {
    pub fn predict(&self, x: f64) -> f64 {
        self.weight * x + self.bias
    }

    pub fn train(&mut self, dataset: &[(f64, f64)], cfg: &Config) -> Vec<(f64, f64)> {
        let dataset_size: f64 = dataset.len() as f64;
        let mut history = Vec::new();

        for epoch in 0..cfg.epochs {
            history.push((epoch as f64, loss(dataset, self)));

            let mut error_x_sum = 0.0;
            let mut error_sum   = 0.0;

            for (x, actual) in dataset {
                let error = self.predict(*x) - actual;
                error_x_sum += error * x;
                error_sum   += error;
            }

            let grad_w = (2.0 / dataset_size) * error_x_sum;
            let grad_b = (2.0 / dataset_size) * error_sum;

            self.weight -= cfg.lr * grad_w;
            self.bias   -= cfg.lr * grad_b;

            if epoch % 100 == 0 {
                println!(
                    "epoch {:>4}  weight: {:>7.4}  bias: {:>7.4}  loss: {:>10.4}",
                    epoch,
                    self.weight,
                    self.bias,
                    loss(dataset, self)
                );
            }
        }

        history
    }

    pub fn train_naive(&mut self, dataset: &[(f64, f64)], cfg: &Config) -> Vec<(f64, f64)> {
        let mut history = Vec::new();

        for epoch in 0..cfg.epochs {
            history.push((epoch as f64, loss(dataset, self)));

            for (x, actual) in dataset {
                let error = self.predict(*x) - actual;
                if error > 0.0 {
                    self.weight -= 0.01;
                    self.bias -= 0.01;
                } else if error < 0.0 {
                    self.weight += 0.01;
                    self.bias += 0.01;
                }
            }

            if epoch % 100 == 0 {
                let loss = loss(dataset, self);
                println!(
                    "epoch {:>4}  weight: {:>7.4}  bias: {:>7.4}  loss: {:>10.4}",
                    epoch, self.weight, self.bias, loss
                );
            }
        }
        history
    }
}

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
