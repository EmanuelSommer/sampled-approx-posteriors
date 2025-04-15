library(ggplot2)
h_function <- function(b, w1, w2, c) {
  dnorm(b) * dnorm(w1) * dnorm(w2) - dnorm(b * c) * dnorm(w1 * c) * dnorm(w2 / c)
}
tibble::tibble(c = seq(0.6, 1.05, length.out = 1000),
                     eq = h_function(b = 0.9, w1 = 0.5, w2 = 0.8, c = c)) |>
  ggplot(aes(x = c, y = eq)) +
  geom_hline(yintercept = 0, color = "grey") +
  geom_line() +
  geom_point(color = "red", data = data.frame(eq=0, c=0.77754)) +
  labs(y="h(c)", x="c") +
  theme_minimal() +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        plot.title = element_text(size = 16, face = "bold"),
        legend.position = "none")
