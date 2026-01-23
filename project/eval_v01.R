suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

today <- Sys.Date()
date <- format(today, "_%m_%d")

# Root folder where eval_haversine_*.csv files exist.
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
details_csv <- file.path(root_dir, "eval_haversine_details.csv")
out_dir <- "D:/LabPCWhite/KNK_Lab/_Projects/R_faiss_stat/plots_eval_haversine"

if (!file.exists(summary_csv)) stop(paste("missing summary:", summary_csv))
if (!file.exists(details_csv)) stop(paste("missing details:", details_csv))

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summary <- fread(summary_csv)
details <- fread(details_csv)

# Ensure consistent group order across plots (by haversine recall desc).
if ("haversine_recall_at_k" %in% names(summary)) {
  summary[, group := factor(group, levels = summary[order(-haversine_recall_at_k), group])]
}
summary[, group_short := tstrsplit(as.character(group), "_", fixed = TRUE)[[1]]]
if ("haversine_recall_at_k" %in% names(summary)) {
  summary[, group_short := factor(group_short, levels = unique(summary[order(-haversine_recall_at_k), group_short]))]
} else {
  summary[, group_short := factor(group_short, levels = unique(group_short))]
}
details[, group_short := tstrsplit(as.character(group), "_", fixed = TRUE)[[1]]]
details[, group_short := factor(group_short, levels = levels(summary$group_short))]

to_long <- function(dt, cols, label_map, value_name = "value", id_var = "group_short") {
  long <- melt(dt, id.vars = id_var, measure.vars = cols,
               variable.name = "metric", value.name = value_name)
  long[, metric := label_map[metric]]
  long
}

plot_save <- function(p, name, w = 9, h = 5) {
  ggsave(file.path(out_dir, paste0(name, ".jpg")), plot = p, width = w, height = h, dpi = 300)
  p
}

metric_theme <- theme_minimal() + theme(axis.text.x = element_text(angle = 30, hjust = 1))

label_unit <- function(unit) {
  function(x) {
    paste0(format(x, scientific = FALSE, trim = TRUE), " ", unit)
  }
}

label_ratio <- function(x) {
  format(x, scientific = FALSE, trim = TRUE)
}

log_breaks <- function(x) {
  x <- x[is.finite(x) & x > 0]
  if (!length(x)) {
    return(NULL)
  }
  low <- floor(log10(min(x)))
  high <- ceiling(log10(max(x)))
  10 ^ (low:high)
}

# 02_haversine_min_topk_box (Haversine Min TopK Distance Distribution)
p_haversine_box <- ggplot(details[!is.na(haversine_min_topk_m)], aes(x = group_short, y = haversine_min_topk_m)) +
  geom_boxplot(outlier.size = 0.5) +
  coord_cartesian(ylim = c(0, 500)) +
  labs(title = "Haversine Min TopK Distance Distribution", x = "model", y = "distance (m)") +
  metric_theme


haversine_hit_cols <- grep("^haversine_hit_", names(summary), value = TRUE)
if (length(haversine_hit_cols) > 0) {
  parse_threshold <- function(col) {
    raw <- sub("^haversine_hit_", "", col)
    raw <- sub("m$", "", raw)
    raw <- gsub("p", ".", raw)
    as.numeric(raw)
  }
  haversine_hit_labels <- setNames(sub("^haversine_hit_", "hit@", haversine_hit_cols), haversine_hit_cols)
  haversine_hit_long <- to_long(summary, haversine_hit_cols, haversine_hit_labels)
  haversine_hit_long[, threshold_m := parse_threshold(sub("^hit@", "", metric))]
  haversine_hit_long <- haversine_hit_long[order(threshold_m)]
  haversine_hit_long <- haversine_hit_long[value > 0]
  
  # 08_haversine_hit_curve(Haversine Hit@K Curve + Area)
  p_haversine_hit_curve <- ggplot(haversine_hit_long, aes(x = threshold_m, y = value, color = group_short, fill = group_short)) +
    geom_ribbon(aes(ymin = 1e-3, ymax = value), alpha = 0.12, color = NA) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    scale_x_continuous(breaks = sort(unique(haversine_hit_long$threshold_m))) +
    scale_y_log10(limits = c(1e-3, 1), breaks = c(1e-3, 1e-2, 1e-1, 1), labels = label_ratio) +
    labs(title = "Haversine Hit@K (curve)", x = "threshold (m)", y = "hit ratio (log10)") +
    theme_minimal() + 
    scale_color_brewer(palette = "Set2", name = "models") +
    scale_fill_brewer(palette = "Set2", name = "models")
}

if ("total_ms" %in% names(details)) {
  latency_mean <- details[!is.na(total_ms), .(mean_ms = mean(total_ms, na.rm = TRUE)), by = group_short]
  p_latency <- ggplot(details[!is.na(total_ms)], aes(x = group_short, y = total_ms, fill = group_short)) +
    geom_violin(alpha = 0.3, trim = FALSE) +
    geom_boxplot(width = 0.2, outlier.size = 0.5) +
    geom_text(
      data = latency_mean,
      aes(x = group_short, y = mean_ms, label = sprintf("mean: %.1f", mean_ms)),
      hjust = 0,
      nudge_x = 0.25,
      size = 3,
      inherit.aes = FALSE
    ) +
    labs(title = "Latency Distribution (Total)", x = "model", y = "latency (ms)") +
    metric_theme + 
    labs(fill = "models") + 
    scale_fill_brewer(palette = "Set2", name = "models")
}

if (all(c("embed_ms_mean", "search_ms_mean") %in% names(summary))) {
  time_long <- melt(
    summary,
    id.vars = "group_short",
    measure.vars = c("embed_ms_mean", "search_ms_mean"),
    variable.name = "component",
    value.name = "ms"
  )
  time_long[, component := factor(component,
                                  levels = c("embed_ms_mean", "search_ms_mean"),
                                  labels = c("embed", "search"))]
  time_wide <- dcast(time_long, group_short ~ component, value.var = "ms")
  time_wide[, embed := fifelse(is.na(embed), 0, embed)]
  time_wide[, search := fifelse(is.na(search), 0, search)]
  if ("total_ms_mean" %in% names(summary)) {
    time_total <- summary[, .(group_short, total_ms_mean)]
  } else {
    time_total <- time_wide[, .(group_short, total_ms_mean = embed + search)]
  }
  time_labels <- merge(
    time_total,
    time_wide[, .(group_short, embed, search)],
    by = "group_short",
    all.x = TRUE
  )
  time_labels[, label := sprintf("total: %.1f\nembed: %.1f\nsearch: %.1f", total_ms_mean, embed, search)]
  label_nudge <- 0.02 * max(time_total$total_ms_mean, na.rm = TRUE)
  p_time_stack <- ggplot(time_long, aes(x = group_short, y = ms, fill = component)) +
    geom_col(position = position_stack(reverse = TRUE)) +
    geom_text(
      data = time_labels,
      aes(x = group_short, y = total_ms_mean, label = label),
      vjust = 0,
      nudge_y = label_nudge,
      size = 3,
      inherit.aes = FALSE
    ) +
    labs(title = "Breakdown Latency Results (Mean)", x = "model", y = "latency (ms)", fill = "stage") +
    metric_theme +
    coord_cartesian(ylim = c(0, 500))
}

if ("total_ms_mean" %in% names(summary) && "haversine_top1_mean_m" %in% names(summary)) {
  tradeoff_top1 <- summary[!is.na(total_ms_mean) & !is.na(haversine_top1_mean_m)]
  tradeoff_top1 <- tradeoff_top1[haversine_top1_mean_m > 0]
  p_tradeoff_top1 <- ggplot(
    tradeoff_top1,
    aes(x = total_ms_mean, y = haversine_top1_mean_m, color = group_short)
  ) +
    geom_point(size = 3) +
    geom_text(
      aes(label = sprintf("%.1f", haversine_top1_mean_m)),
      hjust = 0,
      nudge_x = 0.02 * diff(range(tradeoff_top1$total_ms_mean)),
      size = 3,
      show.legend = FALSE
    ) +
    scale_y_log10(breaks = log_breaks(tradeoff_top1$haversine_top1_mean_m), labels = label_unit("m")) +
    labs(title = "Speed-Accuracy Trade-Off (haversine top1 distance)",
         x = "total latency (ms)", y = "top1 distance (m, log10)") +
    theme_minimal() + 
    labs(color = "models") + 
    scale_color_brewer(palette = "Set2", name = "models")
}

plot_save(p_haversine_box, paste0("02_haversine_min_topk_box", date))
if (exists("haversine_hit_long")) {
  plot_save(p_haversine_hit_curve, paste0("04_haversine_hit_curve", date), w = 9, h = 5)
}
if (exists("p_latency")) {
  plot_save(p_latency, paste0("05_latency_distribution", date), w = 9, h = 5)
}
if (exists("p_time_stack")) {
  plot_save(p_time_stack, paste0("06_latency_breakdown", date), w = 9, h = 5)
}
if (exists("p_tradeoff_top1")) {
  plot_save(p_tradeoff_top1, paste0("08_speed_accuracy_tradeoff_top1", date), w = 9, h = 5)
}

pdf(file.path(out_dir, paste0("eval_haversine_report", date, ".pdf")), w = 9, h = 5)
print(p_haversine_box)
if (exists("haversine_hit_long")) {
  print(p_haversine_hit_curve)
}
if (exists("p_latency")) {
  print(p_latency)
}
if (exists("p_time_stack")) {
  print(p_time_stack)
}
if (exists("p_tradeoff_top1")) {
  print(p_tradeoff_top1)
}
dev.off()

message("[INFO] plots saved to: ", out_dir)
