# Carichiamo le librerie necessarie per l'esecuzione dello script e carichiamo il Dataset che useremo per il modello predittivo

library(readr)
library(car)
library(ggplot2)
library(GGally)
library(e1071)
library(gridExtra)
library(corrplot)
library(glmnet)
library(lmtest)
library(coefplot)
library(dplyr)
library(psych)
dati <- read_csv("bodyfat.csv")

# Iniziamo facendo un analisi delle variabili, con una breve descrizione delle stesse
#Density: Densità corporea determinato dalla pesatura subacquea.
#BodyFat: Percentuale di grasso corporeo dall'Equazione di Siri - Variabile di interesse
#Age: Età
#Weight: Peso (in libbre)
#Height: Altezza (in pollici)
#Neck: Circonferenza del collo
#Chest: Circonferenza del torace
#Abdomen: Circonferenza dell'addome
#Hip: Circonferenza dei fianchi
#Thigh: Circonferenza della coscia
#Knee: Circonferenza del ginocchio
#Ankle: Circonferenza della caviglia
#Biceps: Circonferenza del bicipite
#Forearm: Circonferenza dell'avambraccio
#Wrist: Circonferenza del polso
dati<- dati %>% rename(Densita = Density,Perc_grasso = BodyFat,Eta = Age,Peso = Weight,Altezza = Height,
                       Collo = Neck,Torace = Chest,Addome = Abdomen,Fianchi = Hip,Coscia = Thigh,
                      Ginocchio = Knee,Caviglia = Ankle,Bicipite = Biceps,Avambraccio = Forearm,Polso = Wrist)
# Effettuiamo la trasformazione delle variabili peso e altezza poichè presentano unità di misura differenti
dati$Peso<-(dati$Peso)*0.453592 
dati$Altezza<-(dati$Altezza)*2.54
#decidiamo di eliminare la variabile Density
dati <- dati[,-1]
#abbiamo moltiplicato i dati delle variabili con i rispettivi fattori di conversione, e li abbiamo trasformati rispettivamente in chili e centimetri
# Per Prima cosa controlliamo se nel dataset sono presenti missing
missing_values <- colSums(is.na(dati))
print(missing_values)
# Analisi Descrittiva
summary(dati)
pairs.panels(dati,method = "pearson",hist.col = "#00AFBB", density = TRUE,ellipses = TRUE, lm = TRUE, 
             stars = TRUE,cex.cor = 1.2,font.labels = 2,main = "Analisi Correlazione Pre-Pulizia",gap = 0.5,jiggle = TRUE)
#Dall'analisi grafica della distribuzione delle variabili emerge che alcune di esse sono fortemente Asimmetriche,
#dunque calcoliamo l'asimmetria per queste variabili.

# Calcolo dell'Asimmetria per ogni variabile con grafico

Asimmetria<- apply(dati, 2, skewness) 
#Creazione di un dataframe dalla distorsione per il plotting
dtasimmetria<- data.frame(Variabile = names(Asimmetria), Asimettria = Asimmetria)
ggplot(dtasimmetria, aes(x = reorder(Variabile, Asimmetria), y = Asimmetria, fill = Asimmetria > 0)) +
geom_bar(stat = "identity", color = "black") + geom_text(aes(label = round(Asimmetria, 2), 
vjust = ifelse(Asimmetria > 0, -0.5,1.5)),size = 5) +scale_fill_manual(values = c("TRUE" = "#00AFBB", "FALSE" = "#E7B800"),
labels = c("Negativa", "Positiva")) + geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +theme_minimal() +
theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),legend.position = "bottom",legend.title = element_blank())+
labs(title = 'Coefficiente di Asimmetria (Skewness) delle Variabili', x = 'Variabili',  y = 'Coefficiente di Asimmetria')

#Boxplot
p1 <- ggplot(dati, aes(y = Caviglia , x = "")) + 
  geom_boxplot(fill = "00AFBB", color = "black", outlier.colour = "red", outlier.size = 3) + 
  theme_minimal(base_size = 12) +
  labs(title = "Caviglia", y = "Circonferenza (cm)", x = "")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

p2 <- ggplot(dati, aes(y = Addome, x = "")) + 
  geom_boxplot(fill = "00AFBB", color = "black", outlier.colour = "red", outlier.size = 3) + 
  theme_minimal(base_size = 12) +
  labs(title = "Addome", y = "Circonferenza (cm)", x = "")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

p3 <- ggplot(dati, aes(y = Peso, x = "")) + 
  geom_boxplot(fill = "00AFBB", color = "black", outlier.colour = "red", outlier.size = 3) + 
  theme_minimal(base_size = 12) +
  labs(title = "Peso", y = "Chilogrammi (KG)", x = "")+
 theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

p4 <- ggplot(dati, aes(y = Fianchi, x = "")) + 
  geom_boxplot(fill = "00AFBB", color = "black", outlier.colour = "red", outlier.size = 3) + 
  theme_minimal(base_size = 12) +
  labs(title = "Fianchi", y = "Circonferenza (cm)", x = "")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)

###### FUNZIONE PER TROVARE ED ELIMINARE GLI OUTLIERS BASATO SU INTERVALLI TIPICI
remove_outliers_sd <- function(data, k) {
  # Copia del data frame originale
  clean_data <- data
  
  # Per ogni colonna nel data frame
  for (col in names(data)) {
    if (is.numeric(data[[col]])) {
      # Calcola la media e la deviazione standard
      mean_val <- mean(data[[col]], na.rm = TRUE)
      sd_val <- sd(data[[col]], na.rm = TRUE)
      
      # Definisci i limiti inferiore e superiore
      lower_bound <- mean_val - k * sd_val
      upper_bound <- mean_val + k * sd_val
      
      # Rimuovi le righe con valori fuori dagli intervalli tipici
      clean_data <- clean_data[clean_data[[col]] >= lower_bound & clean_data[[col]] <= upper_bound, ]
    }
  }
  
  return(clean_data)
}
#### k = fattore di scala, in questo caso con k = 3 gli intervalli tipici basati sulla dev std restituiscono 
#### il 99% osservazioni
k=3
clean_dati <- remove_outliers_sd(dati, k)
# Rappresentazione grafica della distribuzione delle variabili che presentavano gli outliers
sp1 <- ggplot(clean_dati, aes(x = Coscia)) + 
  geom_histogram(aes(y=..density..), bins = 40, fill = 'skyblue') +
  geom_density(color = 'blue') +
  ggtitle('Distribuzione di Coscia') +
  theme_minimal()

sp2 <- ggplot(clean_dati, aes(x = Caviglia)) + 
  geom_histogram(aes(y=..density..), bins = 40, fill = 'lightcoral') +
  geom_density(color = 'red') +
  ggtitle('Distribuzione di Caviglia') +
  theme_minimal()

sp3 <- ggplot(clean_dati, aes(x = Peso)) + 
  geom_histogram(aes(y=..density..), bins = 40, fill = 'gold') +
  geom_density(color = 'darkorange') +
  ggtitle('Distribuzione di Peso') +
  theme_minimal()

sp4 <- ggplot(clean_dati, aes(x = Fianchi)) + 
  geom_histogram(aes(y=..density..), bins = 40, fill = 'mediumseagreen') +
  geom_density(color = 'green') +
  ggtitle('Distribuzione di Fianchi') +
  theme_minimal()
grid.arrange(sp1, sp2, sp3, sp4, nrow = 2, ncol = 2)
#Qui si nota come, dopo l'eliminazione, la distribuzione dei valori si avvicini di più ad una Normale rispetto a prima

Asimmetria_2<- apply(clean_dati, 2, skewness)
#Creazione di un dataframe dalla skewness per il plotting
dtasimmetria_2<- data.frame(Variabile = names(Asimmetria_2), Skewness = Asimmetria_2)
ggplot(dtasimmetria_2, aes(x = reorder(Variabile, Asimmetria_2), y = Asimmetria_2, fill = Asimmetria_2 > 0)) +
geom_bar(stat = "identity", color = "black") + geom_text(aes(label = round(Asimmetria_2, 2), 
vjust = ifelse(Asimmetria_2 > 0, -0.5,1.5)),size = 5) +scale_fill_manual(values = c("TRUE" = "#00AFBB", "FALSE" = "#E7B800"),
labels = c("Negativa", "Positiva")) + geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +theme_minimal() +
theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),legend.position = "bottom",legend.title = element_blank())+
labs(title = 'Coefficiente di Asimmetria post pulizia', x = 'Variabili',  y = 'Coefficiente di Asimmetria')

#Dopo Aver pulito il Dataset, passiamo all'analisi della correlazione tra le variabili, per verificare se esiste multicollinearità.
# Calcolo della matrice di correlazione
cor<- cor(clean_dati)
corrplot(cor, method = "color",type = "upper",order = "hclust",
         col =colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(8),addrect = 3,tl.col = "black",
         tl.srt = 45,tl.cex = 1.0,addCoef.col = "black",number.cex = 0.8,diag = FALSE)
#Si può notare, anche senza effettuare i relativi test, che quasi tutte le variabili mostrano correlazioni significative; Osserviamo una correlazione negativa notevole tra la densità e il grasso corporeo, 
#prossima a -1. Allo stesso tempo,il peso manifesta correlazioni positive con diverse variabili. Tuttavia, l'esclusione di tali variabili potrebbe non essere giustificata da un punto di vista scientifico. 
#Pertanto, intendiamo utilizzare Lasso e Ridge Regression per affrontare il problema della elevata correlazione.

#### Calcolo del condition number
subset_regressori <- subset(clean_dati, select = -1)
X <- as.matrix(cbind(rep(1, nrow(clean_dati)), subset_regressori)) # costruisco la matrice disegno
autoval<-eigen(t(X)%*%X) # calcolo di autovalori e autovettori della matrice disegno
max(autoval$values) # max autovalore
min(autoval$values) #min autovalore
condition.number<-sqrt(max(autoval$values)/min(autoval$values)) # condition number
print(condition.number) # notiamo come il condition number ha un valore elevatissimo (di parecchio sopra il 30, considerato come valore di riferimento quantomeno pratico)
# si rende perciò necessario l'utilizzo di tecniche di regolarizzazione per trattare il serio problema della multicollinearità che presentano le var. esplicative del dataset

#Modello Completo
mfull <- lm(Perc_grasso ~  Eta + Peso + Altezza + Collo + Torace + Addome + Fianchi + Coscia + Ginocchio + Caviglia + Bicipite + Avambraccio + Polso, data = clean_dati)
summary(mfull)

### Calcolo del VIF
vif_values <- vif(mfull)
print(vif_values)
#Tollerance
toll_values<-1/vif(mfull)
print(toll_values)
#indice r^2
rquadro<-(vif(mfull)-1)/vif(mfull)
print(rquadro)
#Estrazione dei residui e delle ordinate stimate
res1<-resid(mfull)
fit1<-fitted(mfull)
#Facciamo un plot dei residui per visualizzarlo
plot(fit1,res1)
#Da un analisi grafica preliminare non sembra che i residui abbiano un andamento particolare. Effettuiamo comunque i test per verificarlo.

# TEST DI BREUSCH-PAGAN
#Calcoliamo il quadrato dei residui
res12<-res1^2 
#Modello lineare dei residui al quadrato in funzione dei regressori
modres12<-lm(res12~Eta + Peso + Altezza + Collo + Torace + Addome + Fianchi + Coscia + Ginocchio + Caviglia + Bicipite + Avambraccio + Polso, data = clean_dati)
summary(modres12) #p-value = 0.09153
#Effettuando i test, risulta che i dati sono eteroschedastici. Dovremo dunque effettuare una trasformazione delle variabili per rimediare a questa problematica.
fit1<-fitted(mfull)
fit12<-fit1^2
modresW<-lm(res12~fit1+fit12)
summary(modresW)


# Addestramento del modello predittivo - Train e Test Set

#addestriamo il modello, impostiamo un seme casuale e suddividiamo le osservazioni creando un indice
set.seed(100)
index=sample(1:nrow(clean_dati),0.75*nrow(clean_dati))
trainRW = clean_dati[index,]
testRW = clean_dati[-index,]
dim(trainRW)
dim(testRW)

##Stima del modello con il Train set
mtrainRW<-lm(Perc_grasso ~  Eta + Peso + Altezza + Collo + Torace + Addome + Fianchi + Coscia + Ginocchio + Caviglia + Bicipite + Avambraccio + Polso, data=trainRW)
summary(mtrainRW)

#Definiamo le funzioni MSE e RMSE
mse = function(actual, predicted) {
  mean((actual - predicted) ^ 2) }

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2)) }

# Usiamo le stesse funzioni per calcolare MSE e del RMSE nel test set
(mseTRAIN<-mse(actual = trainRW$Perc_grasso, predicted = predict(mtrainRW, trainRW)))

(rmseTRAIN<-rmse(actual = trainRW$Perc_grasso, predicted = predict(mtrainRW, trainRW)))

(mseTEST<-mse(actual = testRW$Perc_grasso, predicted = predict(mtrainRW, testRW)))

(rmseTEST<-rmse(actual = testRW$Perc_grasso, predicted = predict(mtrainRW, testRW)))

## Applichiamo le tecniche di regolarizzazzione

#Creazione delle matrici di design per addestramento e test
X_train <- subset(trainRW, select = -c(Perc_grasso))
y_train <- trainRW$Perc_grasso
X_test <- subset(testRW, select = -c(Perc_grasso))
y_test <- testRW$Perc_grasso

#Procediamo applicando L'Elastic Net con i vari coefficenti di Alpha, cosi da Testare Ridge Regression, Lasso e la possibile combinazione delle due tecniche 
par(mar = c(5, 4, 7, 2) + 0.1)
#alpha 1 Lasso
bf1<- glmnet(as.matrix(X_train), y_train, alpha =1)
plot(bf1, main="Traccia dei Coefficienti - Lasso (alpha = 1)",xvar="lambda",label=TRUE,lwd=2)
print(coef(bf1))

#alpha 0.5 Elastic net
bf.5<-glmnet(as.matrix(X_train), y_train, alpha=0.5)
plot(bf.5, main="Traccia dei Coefficienti - EN (alpha = 0.5)",xvar="lambda",label=TRUE,lwd=2)
print(coef(bf.5))

#alpha 0.1 Elastic Net
bf.1<-glmnet(as.matrix(X_train), y_train, alpha=0.1)
plot(bf.1, main="Traccia dei Coefficienti - EN (alpha = 0.1)",xvar="lambda",label=TRUE,lwd=2)
print(coef(bf.1))


#alpha 0 ridge
bfr<-glmnet(as.matrix(X_train), y_train, alpha=0)
plot(bfr, main="Traccia dei Coefficienti - Ridge Regression (alpha = 0)",xvar="lambda",label=TRUE,lwd=2)
print(coef(bfr))

#Leave one out Cross Validation
#LOOCV Lasso
loocvl=cv.glmnet(as.matrix(X_train), y_train, nfolds=nrow(X_train) ,grouped=FALSE, alpha=1) 
plot(loocvl,main = "Cross-Validation del Modello Lasso")
(bestLambda.bf1<-loocvl$lambda.min)
loocvl.mod=glmnet(as.matrix(X_train), y_train,alpha=1,lambda=bestLambda.bf1)
coef(loocvl.mod)[,1]

#LOOCV Elastic Net Alpha 0.1
loocv.1=cv.glmnet(as.matrix(X_train), y_train, nfolds=nrow(X_train) ,grouped=FALSE, alpha=0.1) 
plot(loocv.1,,main = "Cross-Validation del Modello EN(0.1)")
(bestLambda.bf.1<-loocv.1$lambda.min)
loocv.1.mod=glmnet(as.matrix(X_train), y_train,alpha=0.1,lambda=bestLambda.bf.1)
coef(loocv.1.mod)[,1]

#LOOCV Ridge
loocvr=cv.glmnet(as.matrix(X_train), y_train, nfolds=nrow(X_train) ,grouped=FALSE, alpha=0) 
plot(loocvr,main = "Cross-Validation del Modello Ridge")
(bestLambda.bfr<-loocvr$lambda.min)
loocvr.mod=glmnet(as.matrix(X_train), y_train,alpha=0,lambda=bestLambda.bfr)
coef(loocvr.mod)[,1]

#LOOCV Alpha 0.5
loocv.5=cv.glmnet(as.matrix(X_train), y_train, nfolds=nrow(X_train) ,grouped=FALSE, alpha=0.5) 
plot(loocv.5,main = "Cross-Validation del Modello EN(0.5)")
(bestLambda.bf.5<-loocv.5$lambda.min)
loocv.5.mod=glmnet(as.matrix(X_train), y_train,alpha=0.5,lambda=bestLambda.bf.5)
coef(loocv.5.mod)[,1]

# Analisi degli output e confronto tra le tecniche utilizzate

#Analizziamo ora i risultati, vedendo quale tecnica minimizza l'mse del train set. Il valore minore corrisponderà al miglior modello predittivo.
(mse.minLASSO <- loocvl$cvm[loocvl$lambda == loocvl$lambda.min])
(mse.minEN05 <- loocv.5$cvm[loocv.5$lambda == loocv.5$lambda.min])
(mse.minEN01 <- loocv.1$cvm[loocv.1$lambda == loocv.1$lambda.min])
(mse.minRR <- loocvr$cvm[loocvr$lambda == loocvr$lambda.min])
min(mse.minLASSO,mse.minEN05,mse.minRR,mse.minEN01)
#Il Metodo che minimizza l'MSE risulta essere l'en 0.5
valori_mse <- c(mse.minLASSO, mse.minRR , mse.minEN05, mse.minEN01)
dataframe_mse <- data.frame(Modello = c("Lasso", "Ridge ", "En Aplha=0.5","En Aplha=0.1"),MSE = valori_mse)
min_mse_value <- min(dataframe_mse$MSE)
ggplot(dataframe_mse,  aes(x = reorder(Modello, MSE), y = MSE, color = (MSE == min_mse_value))) +
       geom_col(aes(fill = (MSE == min_mse_value)), width = 0.5,color = "black") + 
       geom_text(aes(label = round(MSE, 3)), vjust = -0.5, size = 5, color = "black") + 
       scale_fill_manual(values = c("FALSE" = "red", "TRUE" = "green")) +
       scale_color_manual(values = c("FALSE" = "red", "TRUE" = "green")) +
       theme_minimal() +theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
       axis.title = element_text(size = 14, face = "bold"),axis.text.x = element_text(size = 12),
       legend.position = "none" ) +labs(title = "Confronto MSE tramite LOOCV",
       subtitle = "Modelli ordinati per MSE crescente. Il modello Elastic Net (α=0.5) è il vincitore.",
       x = "Modello",y = "Errore Quadratico Medio (MSE)")

#############
coef_matrix <- as.matrix(coef(loocv.5, s = "lambda.min")) 
df_coef <- data.frame(Variabile = rownames(coef_matrix),Valore = coef_matrix[,1]) %>%
  filter(Variabile != "(Intercept)") %>%arrange(desc(abs(Valore)))
ggplot(df_coef, aes(x = reorder(Variabile, Valore), y = Valore, fill = Valore > 0)) +
       geom_bar(stat = "identity", color = "black", width = 0.7) + coord_flip() + 
       scale_fill_manual(values = c("firebrick", "forestgreen"), labels = c("Correlazione Negativa",
      "Correlazione Positiva")) + labs(title = "Importanza Variabili - Elastic Net (Vincitore)",
       subtitle = "Le variabili senza barra (valore 0) sono state eliminate dal modello",
       x = "", y = "Coefficiente Penalizzato",fill = "Direzione") + theme_minimal() +
       theme(legend.position = "bottom",panel.grid.major.y = element_blank() )

# 1. Calcoliamo le previsioni del VERO vincitore sul Test Set
pred_best_en <- predict(loocv.5, s = loocv.5$lambda.min, newx = as.matrix(X_test))
mse_best_en <- mse(actual = y_test, predicted = pred_best_en)
rmse_best_en <- rmse(actual = y_test, predicted = pred_best_en)
cat("RMSE Elastic Net:    ", round(rmse_best_en, 4), "\n")
cat("MSE Elastic Net:    ", round(mse_best_en, 4), "\n")
sd(dati$Perc_grasso) 
