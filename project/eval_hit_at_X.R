suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

today <- Sys.Date()
date <- format(today, "_%m_%d")

# Root folder where eval_haversine_summary.csv exists.
root_candidates <- c(
  "D:/ImgMatching_export/evaluation",
  "/exports/evaluation",
  "D:/ImgMatching_export/dinov3_faiss_match",
  "/exports/dinov3_faiss_match"
)
root_dir <- root_candidates[which(dir.exists(root_candidates))[1]]
if (is.na(root_dir) || root_dir == "") {
  stop("No valid root_dir found in candidates.")
}

summary_csv <- file.path(root_dir, "eval_haversine_summary.csv")
out_dir <- "D:/LabPCWhite/KNK_Lab/_Projects/R_faiss_stat/plots_eval_haversine_hit_at_x"

if (!file.exists(summary_csv)) stop(paste("missing summary:", summary_csv))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summary <- fread(summary_csv)
if (!"total_ms_mean" %in% names(summary)) {
  summary[, total_ms_mean := NA_real_]
}

if ("haversine_recall_at_k" %in% names(summary)) {
  summary[, group := factor(group, levels = summary[order(-haversine_recall_at_k), group])]
}
summary[, group_short := tstrsplit(as.character(group), "_", fixed = TRUE)[[1]]]
if ("haversine_recall_at_k" %in% names(summary)) {
  summary[, group_short := factor(group_short, levels = unique(summary[order(-haversine_recall_at_k), group_short]))]
} else {
  summary[, group_short := factor(group_short, levels = unique(group_short))]
}

plot_save <- function(p, name, w = 9, h = 5) {
  ggsave(file.path(out_dir, paste0(name, ".jpg")), plot = p, width = w, height = h, dpi = 300)
  p
}

parse_threshold <- function(metric) {
  raw <- sub("^haversine_hit_", "", metric)
  raw <- sub("m$", "", raw)
  raw <- gsub("p", ".", raw)
  as.numeric(raw)
}

hit_cols <- grep("^haversine_hit_", names(summary), value = TRUE)
if (length(hit_cols) == 0) {
  stop("No haversine_hit_* columns found in summary.")
}

hit_long <- melt(
  summary,
  id.vars = c("group_short", "total_ms_mean"),
  measure.vars = hit_cols,
  variable.name = "metric",
  value.name = "hit"
)
hit_long[, threshold_m := parse_threshold(metric)]
hit_long <- hit_long[!is.na(threshold_m)]

# 1) Facet trade-off: total latency vs Hit@X by threshold.
tradeoff_long <- hit_long[!is.na(total_ms_mean)]
if (nrow(tradeoff_long)) {
  p_tradeoff_facets <- ggplot(tradeoff_long, aes(x = total_ms_mean, y = hit, color = group_short)) +
    geom_point(size = 3) +
    facet_wrap(~ threshold_m, scales = "fixed") +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Speed-Accuracy Trade-Off by Hit@X",
         x = "total latency (ms)", y = "hit ratio (0-1)", color = "models") +
    theme_minimal() +
    scale_color_brewer(palette = "Set2", name = "models")
  plot_save(p_tradeoff_facets, paste0("01_tradeoff_facets_hit_at_x", date), w = 12, h = 7)
} else {
  message("[WARN] total_ms_mean missing; skip tradeoff facets.")
}

# 2) Heatmap (Hit@X) + latency bar.
p_heatmap <- ggplot(hit_long, aes(x = threshold_m, y = group_short, fill = hit)) +
  geom_tile() +
  scale_x_continuous(breaks = sort(unique(hit_long$threshold_m))) +
  scale_fill_gradient(low = "#f0f3f5", high = "#1565c0", limits = c(0, 1)) +
  labs(title = "Hit@X Heatmap (Haversine)",
       x = "threshold (m)", y = "model", fill = "hit ratio") +
  theme_minimal()
plot_save(p_heatmap, paste0("02_hit_heatmap", date), w = 10, h = 5)

latency_bar <- summary[!is.na(total_ms_mean)]
if (nrow(latency_bar)) {
  p_latency_bar <- ggplot(latency_bar, aes(x = group_short, y = total_ms_mean, fill = group_short)) +
    geom_col() +
    geom_text(aes(label = sprintf("%.1f", total_ms_mean)), vjust = -0.3, size = 3) +
    labs(title = "Total Latency (Mean)", x = "model", y = "total latency (ms)", fill = "models") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set2", name = "models")
  plot_save(p_latency_bar, paste0("02_latency_bar", date), w = 8, h = 5)
} else {
  message("[WARN] total_ms_mean missing; skip latency bar.")
}

# 3) Hit-curve with latency label.
p_hit_curve <- ggplot(hit_long, aes(x = threshold_m, y = hit, color = group_short)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = sort(unique(hit_long$threshold_m)),
                     expand = expansion(mult = c(0.02, 0.12))) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Hit@X Curve (Haversine) + Latency Label",
       x = "threshold (m)", y = "hit ratio (0-1)", color = "models") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2", name = "models")

label_data <- hit_long[!is.na(total_ms_mean),
                       .(threshold_m = max(threshold_m, na.rm = TRUE),
                         hit = hit[which.max(threshold_m)],
                         total_ms_mean = unique(total_ms_mean)),
                       by = group_short]
if (nrow(label_data)) {
  p_hit_curve <- p_hit_curve +
    geom_text(
      data = label_data,
      aes(label = sprintf("ms=%.1f", total_ms_mean), color = group_short),
      hjust = 0,
      size = 3,
      show.legend = FALSE
    )
}
plot_save(p_hit_curve, paste0("03_hit_curve_latency", date), w = 10, h = 5)

message("[INFO] plots saved to: ", out_dir)
