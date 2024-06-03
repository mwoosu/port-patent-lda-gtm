
install.packages("tm")
install.packages("NLP")
install.packages("stringi")
install.packages("topicmodels" )
install.packages("igraph")
install.packages("lda")
install.packages("lsa")
install.packages("textstem")
install.packages("readxl")
install.packages("writexl")
install.packages("topicmodels")
install.packages("ggplot2")
library(tm)
library(stringi)
library(topicmodels)
library(igraph)
library(lda)
library(cluster)
library(NLP)
library(NbClust)
library(grid)
library(MASS)
library(slam)
library(lsa)
library(textstem)
library(readxl)
library(writexl)
library(ggplot2)


setwd("/Users/woosu/Desktop/project")
summary_READ=read.csv("/Users/woosu/Desktop/project/port_patent_data.csv",header = T)
Data=summary_READ[12]
Data=as.matrix(Data) #transform to matrix
dim(Data)


doc_id =c(989:1) 
colnames(Data)=c("text") #열이름 바꾸기
Data= data.frame(doc_id = doc_id, text = Data , stringsAsFactors = FALSE) 

Data.cor=Corpus(DataframeSource(Data))

# 빈 문서를 식별하고 제거
Data.cor <- tm_filter(Data.cor, function(x) {
  return(length(unlist(strsplit(as.character(x), "\\s+"))) > 1)
})

inspect(Data.cor[1])
options(max.print = 9999999)

# 전처리: 특정 단어 보호
Data.cor <- tm_map(Data.cor, content_transformer(protect_words))


# 특수 문자를 직접 제거하는 함수
remove_special_chars <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
Data.cor <- tm_map(Data.cor, remove_special_chars, "“")
Data.cor <- tm_map(Data.cor, remove_special_chars, "”")

#소문자변경
Data.cor <- tm_map(Data.cor, content_transformer(tolower))

#문장부호(구두점)제거
Data.cor <- tm_map(Data.cor, removePunctuation)

#숫자제거
Data.cor <- tm_map(Data.cor, removeNumbers)

#공백제거:tm_map()
Data.cor<-tm_map(Data.cor, stripWhitespace)

#띄어쓰기와 시제 제거
Data.cor <- tm_map(Data.cor, removeWords, stopwords("english"))

#Lemmatization
Data.cor <- lemmatize_words(Data.cor)

# 표제어 추출
Data.cor <- tm_map(Data.cor, content_transformer(lemmatize_strings))

Data.cor <- tm_map(Data.cor, content_transformer(restore_protected_words))
excludes <- c('one', 'first', 'second', stopwords('english'))
Data.cor <- tm_map(Data.cor, removeWords, excludes)
mywords <- readLines("stopwords.txt")
Data.cor<-tm_map(Data.cor, removeWords, mywords)

# DocumentTermMatrix 준비
dtmLDA <- DocumentTermMatrix(Data.cor, control = list(wordLengths = c(2, Inf)))

# 모든 항목이 0인 행 제거
row_sums <- rowSums(as.matrix(dtmLDA))
dtmLDA <- dtmLDA[row_sums > 0, ]

lda_model <- LDA(dtmLDA, k = 6, method = "Gibbs", control = list(
  seed = 26,
  alpha = 0.1,
  delta = 0.05,
  iter = 5000,
  burnin = 1000,
  thin = 500
))

lda_topics <- topics(lda_model, 1) # 각 문서에 대한 주제
lda_terms <- terms(lda_model, 9)   # 각 주제에 대한 상위 9개 용어

# print(lda_topics)
print(lda_terms)

# 문서별 토픽 할당 추출
topic_assignments <- apply(posterior(lda_model)$topics, 1, which.max)

# LDA 토픽 분포 추출
topic_distributions <- posterior(lda_model)$topics

# 데이터 프레임 생성
assignments_df <- data.frame(doc_id = 1:length(topic_assignments), topic = topic_assignments)
distributions_df <- data.frame(doc_id = 1:nrow(topic_distributions), topic_distributions)

# 파일로 저장
write.csv(assignments_df, "document_topic_assignments_k6.csv", row.names = FALSE)
write.csv(distributions_df, "topic_distributions_k6.csv", row.names = FALSE)
