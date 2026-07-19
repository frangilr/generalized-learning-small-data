library(readr)
library(purrr)
library(dplyr)
library(stringr)
library(ggplot2)
library(ggdist)
library(RColorBrewer)

active_key <- "rgb"

postfix_dict <- c(
  "rgb"    = ".csv$"
)

path_dict <- c(
  "rgb"    = "RGB_hist/"
)

path_to_csvs <- paste0("../../data/PanelA/", path_dict[active_key])
file_pattern <- paste0("^all2all_corr_bysubj_forRplots_.*", postfix_dict[active_key])

category_csv_files <- list.files(
  path = path_to_csvs,
  pattern = file_pattern,
  full.names = TRUE 
)

if (length(category_csv_files) != 8) {
  stop(paste("Expected 8 INFANT files, but found", length(category_csv_files), "matching the pattern."))
}

# these names will be used for the facet titles (e.g., "airplane", "bottle")
# removes the "category_" prefix and ".csv" suffix
# can change all2all to all2all and so on... make sure to change all ocurrences
category_names_for_ids <- str_remove(basename(category_csv_files), postfix_dict[active_key]) %>% 
                          str_remove("all2all_corr_bysubj_forRplots_")
names(category_csv_files) <- category_names_for_ids # Name the file path vector

# map_dfr reads each file and binds them into one data frame.
# .id = "category_name" creates a new column using the names we just assigned.
if (active_key != "rgb") {
  column_names <- c("subj", "instance_i", "instance_j", "category", "valid", "hist_corr", "cos_sim")
} else {
  column_names <- c("subj", "instance_i", "instance_j", "category", "valid", "hist_corr")
}

all_categories_data <- map_dfr(
  category_csv_files, 
  ~read_csv(.x, col_names = column_names), 
  .id = "category_name"
)

# this can have duplicates once categories are combined (e.g. an image with a bottle and a bowl will be counted for each)
real_data_dup <- all_categories_data %>%
  filter(valid == TRUE) %>% # valid == TRUE just means within-category similarities (dummy column atm)
  mutate(distribution = "Infant")

# drop duplicates if wanted (unordered pairs)
real_data <- real_data_dup %>%
  filter(!duplicated(cbind(pmin(instance_i, instance_j), pmax(instance_i, instance_j)))) #%>%

########################## read in null ##########################

file_pattern <- paste0("^NULL_forRplots_.*", postfix_dict[active_key])

category_csv_files <- list.files(
  path = path_to_csvs,
  pattern = file_pattern,
  full.names = TRUE 
)

# remove shuffled null files
category_csv_files <- category_csv_files[!grepl("shuffled", category_csv_files)]

if (length(category_csv_files) != 8) {
  stop(paste("Expected 8 NULL files, but found", length(category_csv_files), "matching the pattern."))
}

# these names will be used for the facet titles (e.g., "airplane", "bottle")
# removes the "category_" prefix and ".csv" suffix
# can change all2all to all2all and so on... make sure to change all ocurrences
category_names_for_ids <- str_remove(basename(category_csv_files), postfix_dict[active_key]) %>% 
                          str_remove("NULL_forRplots_")
names(category_csv_files) <- category_names_for_ids # Name the file path vector


# map_dfr reads each file and binds them into one data frame.
# .id = "category_name" creates a new column using the names we just assigned.

all_categories_data_null <- map_dfr(
  category_csv_files, 
  ~read_csv(.x, col_names = column_names), 
  .id = "category_name"
)

null_data_dup <- all_categories_data_null %>%
  filter(valid == TRUE) %>% # valid == TRUE just means within-category similarities (dummy column atm)
  mutate(distribution = "Null")

null_data <- null_data_dup %>%
  filter(!duplicated(cbind(pmin(instance_i, instance_j), pmax(instance_i, instance_j)))) #%>%

# combine datasets for plotting
print(dim(real_data_dup))
print(dim(null_data_dup))
print(dim(real_data))
print(dim(null_data))
aggregated_plot_data <- bind_rows(real_data, null_data)
print(dim(aggregated_plot_data))
target_col <- if (active_key != "rgb") "cos_sim" else "hist_corr"

print(
  ks.test(
    real_data[[target_col]], 
    null_data[[target_col]], 
    alternative='less'
    )
)

print("------") 

print(
  ks.test(
    na.omit(real_data[[target_col]]), 
    na.omit(null_data[[target_col]]), 
    alternative = 'less'
    )
  )

aggregated_plot_data$distribution <- factor(
  aggregated_plot_data$distribution,
  levels = c("Infant", "Null")
)

# with duplicates if wanted
aggregated_plot_data_dup <- bind_rows(real_data_dup, null_data_dup)
aggregated_plot_data_dup$distribution <- factor(
  aggregated_plot_data_dup$distribution,
  levels = c("Infant", "Null")
)
print(dim(aggregated_plot_data_dup))
display.brewer.pal(8, "Set2")

# Pick sBYOL_block_group_3_cific indices
comparison_colors <- c(
  "Infant" = brewer.pal(8, "Set2")[4],
  "Null"   = "grey70"
    # brewer.pal(8, "Set2")[5]
)

aggregated_overlaid_plot <- ggplot(aggregated_plot_data,
                                   aes(y = .data[[target_col]],
                                    #    fill = distribution,
                                    #    color = distribution,
                                       group = distribution
                                       )) +
      ggdist::stat_halfeye(
              aes(fill = distribution),
              position = position_nudge(x = .05, y = 0),
              ## custom bandwidth
              adjust = .7,
              ## adjust height
              width = .8,
              ## move geom to the right
              justification = -.2,
              ## remove slab interval
              .width = 0,
              point_colour = NA,
              alpha = 1.
      ) +

      geom_boxplot(
        aes(fill = distribution, color = distribution, x = .07),
        # width = 0.25, 
        # outlier.shape = NA,
        width = .12,
        alpha = 0.4, 
        # show.legend = FALSE
      ) +
  
  scale_fill_manual(values = comparison_colors, #name = "Distribution",
                      labels = c("Infant", "Null")) +
  scale_color_manual(values = comparison_colors, #name = "Distribution",
                     labels = c("Infant", "Null")) +
  labs(
        title = NULL,
        y = NULL,
        fill = "Distribution", color = "Distribution"
      ) +
      scale_x_discrete(expand = expansion(mult = c(.1, 0))) +
      scale_y_continuous(labels = prettyNum) +
  theme_minimal() +
  theme(
    legend.position = "none",  # Adjust X and Y as needed
    axis.title.x = element_text(size = 22),
    axis.title.y = element_text(size = 22),
    axis.text.x  = element_text(size = 32),
    axis.text.y  = element_text(size = 32),
    legend.title = element_text(size = 20),
    legend.text  = element_text(size = 18),
    strip.text   = element_text(size = 20),
    axis.line = element_line(colour="grey60"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank()
  ) +
      coord_flip(ylim = c(0., 1.0))
      # coord_flip(ylim = c(0., 5.))  # just for GIST

pdf_width <- 9 
pdf_height <- 6

## already saved but can uncomment
##
if (!is.null(aggregated_overlaid_plot)) {
    output_filename <- paste0("rainplot_infant_vs_null_", active_key, ".svg")
    ggsave(
      filename = output_filename, 
      plot = aggregated_overlaid_plot, 
      width = pdf_width, 
      height = pdf_height, 
      dpi = 300, 
      device = "svg"
    )
  } else {
    message("Plot generation failed")
  }

print(active_key)

# ############################################################################################################################################################

# # plot for entropy (SI)

# ############################################################################################################################################################

library(RColorBrewer)
library(ggplot2)
library(ggnewscale)
library(entropy)

all_vals <- na.omit(aggregated_plot_data_dup$hist_corr)
# discretize
breaks <- seq(min(all_vals), max(all_vals), length.out = 101)  # 100 bins

# each category as a dot
entropy_data <- aggregated_plot_data_dup %>%
  group_by(distribution, category_name) %>%
  summarise(entropy_dist = {
    vals <- na.omit(hist_corr)
    if(length(vals) > 1) {
      counts <- hist(vals, breaks = breaks, plot = FALSE)$counts
      entropy(counts, unit = "log2")
    } else {
      NA_real_
    }
  }, .groups = 'drop')

# all categories
entropy_summary <- aggregated_plot_data %>%
  group_by(distribution) %>%
  summarise(entropy_dist_m = {
    vals <- na.omit(hist_corr)
    if(length(vals) > 1) {
      counts <- hist(vals, breaks = breaks, plot = FALSE)$counts
      entropy(counts, unit = "log2")
    } else {
      NA_real_
    }
  }, .groups = 'drop')


color_map <- c(
  "Infant" = brewer.pal(8, "Set2")[4],  # 2nd color
  "Null"   = brewer.pal(8, "Set2")[5]   # 5th color
)

bar_order <- c("Null", "Infant")
entropy_summary$distribution <- factor(entropy_summary$distribution, levels = bar_order)
entropy_data$distribution <- factor(entropy_data$distribution, levels = bar_order)
print(entropy_summary)
safe_colorblind_palette <- c("#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
                           "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888")

entropy_plot <- ggplot(entropy_summary, aes(x = distribution, y = entropy_dist_m, fill = distribution, group = distribution)) +
  geom_col(position = position_dodge(width = 0.9), alpha = 1., width = .8) +

  scale_fill_manual(name = "Distribution", values = color_map, labels = c("Infant", "Null")) +
  new_scale_color() +
  geom_jitter(
    inherit.aes = FALSE,
    data = entropy_data,
    aes(x = distribution, y = entropy_dist, color = category_name, group = distribution),
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9),
    size = 5
  ) +
  scale_color_manual(name = "Category", values = safe_colorblind_palette) +
  labs(
    title = NULL,
    y = NULL,
    x = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",  # Adjust X and Y as needed
    axis.title.x = element_text(size = 24),
    axis.title.y = element_text(size = 24),
    axis.text.x  = element_text(size = 24),
    axis.text.y  = element_text(size = 24),
    legend.title = element_text(size = 22),
    legend.text  = element_text(size = 20),
    strip.text   = element_text(size = 16),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_line(color = "grey92"),
    panel.grid.minor.y = element_line(color = "grey95")
  ) +
  coord_cartesian(ylim = c(6, 6.75))

# if (!is.null(entropy_plot)) {
#     ggsave("entropy_rgb.svg", plot = entropy_plot, width = 9, height = 6, dpi=300, device = "svg")
#   } else {
#     message("Plot generation failed")
#   }