# quick script to set up the data file if not present

library(tidyverse)
fs::dir_create("data")
write_csv(
    diamonds, 
    "data/diamonds.csv"
)
