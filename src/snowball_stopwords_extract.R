load('asset/data_stopwords_snowball.rda')
ls()

View(data_stopwords_snowball)
View(data_stopwords_snowball[["en"]])

# Assuming data_stopwords_snowball[["en"]] is a vector of stopwords
stopwords_en <- data_stopwords_snowball[["en"]]

# Export to TXT file (each stopword on a new line)
writeLines(stopwords_en, "asset/stopwords_en.txt")
