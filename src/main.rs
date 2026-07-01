mod neuron;
mod plots;

fn main() {
    let dataset: Vec<(f64, f64)> = vec![
        (10.0, 28.0),
        (20.0, 38.0),
        (30.0, 47.0),
        (40.0, 56.0),
        (50.0, 64.0),
        (60.0, 74.0),
        (70.0, 82.0),
        (80.0, 91.0),
        (90.0, 100.0),
    ];

    let cfg = neuron::Config {
        epochs: 1000,
        lr: 0.0003,
        w0: 5.0, // weight
        b0: 5.0, // bias
    };

    println!("=== ±0.01 ===\n");
    let mut naive = neuron::Neuron {
        weight: cfg.w0,
        bias: cfg.b0,
    };
    println!("loss antes: {:.4}", neuron::loss(&dataset, &naive));
    let history_naive = naive.train_naive(&dataset, &cfg);
    println!("loss depois: {:.4}\n", neuron::loss(&dataset, &naive));

    println!("=== Gradient Descent (lr={}) ===\n", cfg.lr);
    let mut gd = neuron::Neuron {
        weight: cfg.w0,
        bias: cfg.b0,
    };
    println!("loss antes: {:.4}", neuron::loss(&dataset, &gd));
    let history_gd = gd.train(&dataset, &cfg);
    println!("loss depois: {:.4}\n", neuron::loss(&dataset, &gd));

    println!("--- resultado ---\n");
    println!(
        "{:<10} {:<10} {:<14} {:<14}",
        "x", "real", "±0.01", "gradient"
    );
    println!("{}", "-".repeat(50));
    for (x, actual) in &dataset {
        let p_naive = naive.predict(*x);
        let p_gd = gd.predict(*x);
        println!(
            "{:<10.1} {:<10.1} {:<14.1} {:<14.1}",
            x, actual, p_naive, p_gd
        );
    }

    plots::plot_comparison(
        &dataset,
        (naive.weight, naive.bias),
        (gd.weight, gd.bias),
        &cfg,
        "assets/02_comparison.png",
    );
    plots::plot_loss_comparison(history_naive, history_gd, "assets/02_loss_comparison.png");
    plots::plot_path_on_parabola(&dataset, "assets/02_path.png");
    plots::plot_parabola(&dataset, "assets/02_parabola.png");
    plots::plot_surface_png(&dataset, "assets/02_surface.png");
}
