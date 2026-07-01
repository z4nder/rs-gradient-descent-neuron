use crate::neuron::{Config, Neuron, loss};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

const BG: RGBColor = RGBColor(8, 6, 24);
const PANEL_BG: RGBColor = RGBColor(20, 10, 46);
const GRID: RGBColor = RGBColor(92, 46, 160);
const TEXT: RGBColor = RGBColor(244, 239, 231);
const PURPLE: RGBColor = RGBColor(164, 82, 255);
const GREEN_NEON: RGBColor = RGBColor(181, 223, 0);
const ORANGE_FIRE: RGBColor = RGBColor(255, 120, 0);
const GOLD: RGBColor = RGBColor(255, 180, 50);

fn styled_root(
    path: &str,
    size: (u32, u32),
) -> DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> {
    let root = BitMapBackend::new(path, size).into_drawing_area();
    root.fill(&BG).unwrap();
    root
}

fn paint_panel(area: &DrawingArea<BitMapBackend, plotters::coord::Shift>) {
    area.fill(&PANEL_BG).unwrap();
}

fn style_mesh<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_desc: &str,
    y_desc: &str,
) {
    chart
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc(x_desc)
        .y_desc(y_desc)
        .draw()
        .unwrap();
}

fn style_legend<'a, DB: DrawingBackend + 'a>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) {
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(TEXT.mix(0.35))
        .background_style(BG.mix(0.9))
        .label_font(("sans-serif", 12).into_font().color(&TEXT))
        .draw()
        .unwrap();
}

fn draw_fit_with_errors(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    dataset: &[(f64, f64)],
    w: f64,
    b: f64,
    title: &str,
    line_color: RGBColor,
    error_color: RGBColor,
) {
    paint_panel(area);
    let loss = loss(dataset, &Neuron { weight: w, bias: b });

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 14).into_font().color(&TEXT))
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(0f64..100f64, -10f64..120f64)
        .unwrap();

    style_mesh(&mut chart, "energia", "distância");

    let line: Vec<(f64, f64)> = (0..=100).map(|x| (x as f64, w * x as f64 + b)).collect();
    chart
        .draw_series(LineSeries::new(line, line_color.stroke_width(3)))
        .unwrap()
        .label(format!(
            "previsão  w={:.2}, b={:.2}  Loss={:.0}",
            w, b, loss
        ))
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 24, y)], line_color.stroke_width(3))
        });

    for (x, actual) in dataset {
        let predicted = w * x + b;
        let error = predicted - actual;
        let mid_y = (predicted + actual) / 2.0;

        chart
            .draw_series(LineSeries::new(
                vec![(*x, predicted), (*x, *actual)],
                error_color.mix(0.7).stroke_width(2),
            ))
            .unwrap();

        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:.1}", error),
                (*x + 1.5, mid_y),
                ("sans-serif", 10).into_font().color(&error_color),
            )))
            .unwrap();
    }

    chart
        .draw_series(
            dataset
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 6, GREEN_NEON.filled())),
        )
        .unwrap()
        .label("valores reais")
        .legend(|(x, y)| Circle::new((x + 10, y), 6, GREEN_NEON.filled()));

    chart
        .draw_series(std::iter::once(Circle::new((0.0, -999.0), 0, TRANSPARENT)))
        .unwrap()
        .label("erro")
        .legend(move |(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 24, y)],
                error_color.mix(0.7).stroke_width(2),
            )
        });

    // MSE destacado no canto
    chart
        .draw_series(std::iter::once(Text::new(
            format!("Loss = {:.1}", loss),
            (3.0, 108.0),
            ("sans-serif", 18).into_font().color(&line_color),
        )))
        .unwrap();

    style_legend(&mut chart);
}

// comparação: ±0.01 vs gradient descent — cada um no seu painel com erros explícitos
pub fn plot_comparison(
    dataset: &[(f64, f64)],
    naive: (f64, f64), // (weight, bias) do ±0.01 treinado na main
    gd: (f64, f64),    // (weight, bias) do gradient descent treinado na main
    cfg: &Config,
    path: &str,
) {
    let root = styled_root(path, (1400, 550));
    let (left, right) = root.split_horizontally(700);

    draw_fit_with_errors(
        &left,
        dataset,
        naive.0,
        naive.1,
        &format!("±0.01 — {} epochs", cfg.epochs),
        PURPLE,
        ORANGE_FIRE,
    );
    draw_fit_with_errors(
        &right,
        dataset,
        gd.0,
        gd.1,
        &format!("Gradient Descent — {} epochs  lr={}", cfg.epochs, cfg.lr),
        GOLD,
        GREEN_NEON,
    );

    root.present().unwrap();
    println!("Salvo em {path}");
}

// caminho na parábola — onde cada algoritmo está em cada epoch
pub fn plot_path_on_parabola(dataset: &[(f64, f64)], path: &str) {
    let root = styled_root(path, (1400, 550));
    let (left, right) = root.split_horizontally(700);

    let b_fixed = 18.0;
    let epochs = 50;
    let lr = 0.0001;
    let w0 = 3.0;

    // parábola de fundo
    let parabola: Vec<(f64, f64)> = (0..=300)
        .map(|i| {
            let w = i as f64 * 0.01;
            (
                w,
                loss(
                    dataset,
                    &Neuron {
                        weight: w,
                        bias: b_fixed,
                    },
                ),
            )
        })
        .collect();

    let max_mse = parabola
        .iter()
        .map(|(_, m)| *m)
        .fold(f64::NEG_INFINITY, f64::max);

    // caminho do naive — w a cada epoch
    let mut w_n = w0;
    let mut path_naive: Vec<(f64, f64)> = Vec::new();
    for _ in 0..epochs {
        path_naive.push((
            w_n,
            loss(
                dataset,
                &Neuron {
                    weight: w_n,
                    bias: b_fixed,
                },
            ),
        ));
        for (x, actual) in dataset {
            let error = w_n * x + b_fixed - actual;
            if error > 0.0 {
                w_n -= 0.01;
            } else if error < 0.0 {
                w_n += 0.01;
            }
        }
    }

    // caminho do gradient — w a cada epoch
    let mut w_g = w0;
    let n = dataset.len() as f64;
    let mut path_grad: Vec<(f64, f64)> = Vec::new();
    for _ in 0..epochs {
        path_grad.push((
            w_g,
            loss(
                dataset,
                &Neuron {
                    weight: w_g,
                    bias: b_fixed,
                },
            ),
        ));
        let mut grad_w = 0.0;
        for (x, actual) in dataset {
            let error = w_g * x + b_fixed - actual;
            grad_w += error * x;
        }
        w_g -= lr * (2.0 / n) * grad_w;
    }

    let draw_panel = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                      path_points: &[(f64, f64)],
                      title: &str,
                      dot_color: RGBColor| {
        paint_panel(area);

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 15).into_font().color(&TEXT))
            .margin(28)
            .x_label_area_size(42)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..3.2f64, 0f64..max_mse * 1.1)
            .unwrap();

        style_mesh(&mut chart, "w (weight)", "loss");

        chart
            .draw_series(LineSeries::new(
                parabola.clone(),
                PURPLE.mix(0.5).stroke_width(2),
            ))
            .unwrap()
            .label("loss(w)")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.mix(0.5).stroke_width(2))
            });

        // linha conectando os pontos
        chart
            .draw_series(LineSeries::new(
                path_points.to_vec(),
                dot_color.mix(0.5).stroke_width(1),
            ))
            .unwrap();

        // pontos com numeração a cada 10 epochs
        for (i, (w, m)) in path_points.iter().enumerate() {
            let size = if i == 0 || i == path_points.len() - 1 {
                8
            } else {
                4
            };
            chart
                .draw_series(std::iter::once(Circle::new(
                    (*w, *m),
                    size,
                    dot_color.filled(),
                )))
                .unwrap();

            if i % 10 == 0 {
                chart
                    .draw_series(std::iter::once(Text::new(
                        format!("e{}", i),
                        (*w + 0.03, *m + max_mse * 0.02),
                        ("sans-serif", 11).into_font().color(&dot_color),
                    )))
                    .unwrap();
            }
        }

        chart
            .draw_series(std::iter::once(Circle::new((0.0, -999.0), 0, TRANSPARENT)))
            .unwrap()
            .label("cada ponto = 1 epoch")
            .legend(move |(x, y)| Circle::new((x + 10, y), 5, dot_color.filled()));

        style_legend(&mut chart);
    };

    draw_panel(
        &left,
        &path_naive,
        "Caminho do ±0.01 na parábola — passos fixos",
        ORANGE_FIRE,
    );
    draw_panel(
        &right,
        &path_grad,
        "Caminho do Gradient Descent — passos proporcionais",
        GOLD,
    );

    root.present().unwrap();
    println!("Salvo em {path}");
}

// gráfico — parábola da MSE variando w (b fixo)
pub fn plot_parabola(dataset: &[(f64, f64)], path: &str) {
    let root = styled_root(path, (800, 500));
    paint_panel(&root);

    let b_fixed = 18.0;
    let points: Vec<(f64, f64)> = (0..=300)
        .map(|i| {
            let w = i as f64 * 0.01;
            let n = Neuron {
                weight: w,
                bias: b_fixed,
            };
            (w, loss(dataset, &n))
        })
        .collect();

    let best_w = points
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let max_mse = points
        .iter()
        .map(|(_, m)| *m)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Loss por valor de w — b fixo em 18",
            ("sans-serif", 16).into_font().color(&TEXT),
        )
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..3.0f64, 0f64..max_mse * 1.1)
        .unwrap();

    style_mesh(&mut chart, "w (weight)", "loss");

    chart
        .draw_series(LineSeries::new(points.clone(), PURPLE.stroke_width(3)))
        .unwrap()
        .label("Loss(w)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.stroke_width(3)));

    chart
        .draw_series(std::iter::once(Circle::new(
            (best_w.0, best_w.1),
            8,
            GOLD.filled(),
        )))
        .unwrap()
        .label(format!("mínimo — w={:.2}  Loss={:.0}", best_w.0, best_w.1))
        .legend(|(x, y)| Circle::new((x + 10, y), 6, GOLD.filled()));

    chart
        .draw_series(std::iter::once(Text::new(
            format!("w={:.2}", best_w.0),
            (best_w.0 + 0.05, best_w.1 + max_mse * 0.03),
            ("sans-serif", 13).into_font().color(&GOLD),
        )))
        .unwrap();

    style_legend(&mut chart);
    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_surface_png(dataset: &[(f64, f64)], path: &str) {
    let root = BitMapBackend::new(path, (800, 620)).into_drawing_area();
    root.fill(&BG).unwrap();

    let w_max = 3.0f64;
    let b_max = 30.0f64;
    let steps = 28usize;

    let mut min_loss = f64::INFINITY;
    let mut max_loss = f64::NEG_INFINITY;
    let mut best_w = 0.0f64;
    let mut best_b = 0.0f64;
    for wi in 0..=steps {
        for bi in 0..=steps {
            let w = wi as f64 * w_max / steps as f64;
            let b = bi as f64 * b_max / steps as f64;
            let l = loss(dataset, &Neuron { weight: w, bias: b });
            if l < min_loss {
                min_loss = l;
                best_w = w;
                best_b = b;
            }
            if l > max_loss {
                max_loss = l;
            }
        }
    }
    let best_loss = loss(
        dataset,
        &Neuron {
            weight: best_w,
            bias: best_b,
        },
    );

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Loss(w, b) — parabolóide   mínimo: w={:.2}  b={:.2}  loss={:.1}",
                best_w, best_b, best_loss
            ),
            ("sans-serif", 14).into_font().color(&TEXT),
        )
        .margin(10)
        .build_cartesian_3d(0f64..w_max, min_loss..max_loss, 0f64..b_max)
        .unwrap();

    chart.with_projection(|mut pb| {
        pb.yaw = 0.6;
        pb.pitch = 0.4;
        pb.scale = 0.7;
        pb.into_matrix()
    });

    chart
        .configure_axes()
        .light_grid_style(GRID.mix(0.2))
        .bold_grid_style(GRID.mix(0.5))
        .x_labels(4)
        .y_labels(4)
        .z_labels(4)
        .label_style(("sans-serif", 11).into_font().color(&TEXT))
        .x_formatter(&|x| format!("w={:.1}", x))
        .y_formatter(&|y| format!("{:.0}", y))
        .z_formatter(&|z| format!("b={:.0}", z))
        .draw()
        .unwrap();

    chart
        .draw_series(
            SurfaceSeries::xoz(
                (0..=steps).map(|i| i as f64 * w_max / steps as f64),
                (0..=steps).map(|i| i as f64 * b_max / steps as f64),
                |w, b| loss(dataset, &Neuron { weight: w, bias: b }),
            )
            .style_func(&|&y| {
                let t = ((y - min_loss) / (max_loss - min_loss)).clamp(0.0, 1.0);
                RGBColor(
                    (t * 200.0) as u8,
                    (50.0 + (1.0 - t) * 30.0) as u8,
                    (255.0 - t * 100.0) as u8,
                )
                .mix(0.8)
                .filled()
            }),
        )
        .unwrap();

    let arm_w = w_max * 0.06;
    let arm_b = b_max * 0.06;
    chart
        .draw_series(LineSeries::new(
            vec![
                (best_w - arm_w, best_loss, best_b),
                (best_w + arm_w, best_loss, best_b),
            ],
            GOLD.stroke_width(3),
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            vec![
                (best_w, best_loss, best_b - arm_b),
                (best_w, best_loss, best_b + arm_b),
            ],
            GOLD.stroke_width(3),
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            vec![(best_w, min_loss, best_b), (best_w, best_loss, best_b)],
            GOLD.mix(0.35).stroke_width(1),
        ))
        .unwrap();

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_loss_comparison(
    history_naive: Vec<(f64, f64)>,
    history_gd: Vec<(f64, f64)>,
    path: &str,
) {
    let max_loss = history_naive
        .iter()
        .chain(history_gd.iter())
        .map(|(_, l)| *l)
        .fold(f64::NEG_INFINITY, f64::max);

    let n_epochs = history_naive.len() as f64;
    let loss_naive_end = history_naive.last().map(|(_, l)| *l).unwrap_or(0.0);
    let loss_gd_end = history_gd.last().map(|(_, l)| *l).unwrap_or(0.0);

    let root = styled_root(path, (900, 520));
    paint_panel(&root);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Loss ao longo do treino — ±0.01 vs Gradient Descent",
            ("sans-serif", 16).into_font().color(&TEXT),
        )
        .margin(30)
        .x_label_area_size(42)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..n_epochs, (1f64..max_loss * 1.1).log_scale())
        .unwrap();

    chart
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc("epoch")
        .y_desc("loss (escala log)")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(history_naive, PURPLE.stroke_width(3)))
        .unwrap()
        .label(format!("±0.01  →  loss final: {:.1}", loss_naive_end))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.stroke_width(3)));

    chart
        .draw_series(LineSeries::new(history_gd, GOLD.stroke_width(3)))
        .unwrap()
        .label(format!(
            "Gradient Descent  →  loss final: {:.1}",
            loss_gd_end
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], GOLD.stroke_width(3)));

    // pontos finais
    chart
        .draw_series(std::iter::once(Circle::new(
            (n_epochs - 1.0, loss_naive_end),
            7,
            PURPLE.filled(),
        )))
        .unwrap();
    chart
        .draw_series(std::iter::once(Circle::new(
            (n_epochs - 1.0, loss_gd_end),
            7,
            GOLD.filled(),
        )))
        .unwrap();

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(TEXT.mix(0.35))
        .background_style(BG.mix(0.9))
        .label_font(("sans-serif", 12).into_font().color(&TEXT))
        .draw()
        .unwrap();
    root.present().unwrap();
    println!("Salvo em {path}");
}
