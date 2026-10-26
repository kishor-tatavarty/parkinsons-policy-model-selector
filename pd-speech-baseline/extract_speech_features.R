# Modified for this framework: adapted speech-feature extraction; see project documentation.
##########################################################################
# R code to calculate spectral feature vectors from the UAMS audio files #
##########################################################################

# Copyright (C) 2026 University of Arkansas for Medical Sciences
# Author: Yasir Rahmatallah, yrahmatallah@uams.edu
# Licensed under the Apache License, Version 2.0
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Code was tested using R version 4.4.2, and the following package versions:
# tuneR_1.4.7, gsignal_0.3-7, seewave_2.2.3, signal_1.8-1                   

library(tuneR)
library(gsignal)
library(seewave)

# function to trim silence parts in the audio files
# Note: The used thresholds were determined empirically using the audio 
# files in the UAMS dataset and may require change for a different dataset  
trim_ends <- function(x, w=100, thr=1){
len <- length(x)
env <- numeric(length=len-w)
for(k in 1:(len-w))	env[k] <- sum(x[k:(k+w)]^2)
ind <- which(env > thr)
st <- ind[1]
en <- tail(ind, n=1)
return(c(st, en))
}

# function to calculate linear prediction coding (LPC) coefficients
LPC <- function (wave, f, channel = 1, wl = 256, ovlp = 0, p = 10) 
{
    input <- inputw(wave = wave, f = f, channel = channel)
    wave <- input$w
    f <- input$f
    rm(input)
    n <- nrow(wave)
    lpc <- function(x, p) {
	res <- gsignal::aryule(x=x, p=p)
	return(res)
    }
    if (is.null(wl)) {
        results <- lpc(wave, p)
        return(results)
    }
    else {
        step <- seq(1, n + 1 - wl, wl - (ovlp * wl))
        res <- apply(as.matrix(step), MARGIN = 1, function(x) lpc(wave[x:(wl + x - 1)], p))
        return(res)
    }
}

time.limit <- 1.5 # minimum voiced segment length in seconds
seg.size <- 256 # sliding window width (in samples)
ovlp <- 0.5 # overlap size of the sliding window
p <- 10 # order of the linear prediction model
file_audio_path <- "data/extracted"

fL <- list.files(path=file_audio_path, pattern="[.]wav$", recursive=TRUE, full.names=TRUE, ignore.case=TRUE)
stopifnot(length(fL) > 0)
n.files <- length(fL)
res <- list()
counter <- 0

for(file.name in fL)
	{
	counter <- counter + 1
	file <- file.name
	train_audio <- readWave(file)
	audio.nor <- tuneR::normalize(train_audio, unit="32", pcm=FALSE)
	x <- audio.nor@left # normalized speech signal (mono, not stereo)
	fs <- train_audio@samp.rate
	duration <- length(x) / fs
	ends <- trim_ends(x, w=100, thr=1)

	if((ends[2] - ends[1]) < (fs*time.limit)) 
	{
	print(paste("File ", file.name, " is < ", time.limit," sec, skipped!", sep=""), quote=F)
	next
	} else x <- x[ends[1]:ends[2]]

	# calculate LPC & LAR coefficients
	res.LPC <- LPC(wave=x, f=fs, channel=1, wl=seg.size, ovlp=ovlp, p=p)
	mat.LPC <- mat.LAR <- matrix(0, p, length(res.LPC))
	for(k in 1:length(res.LPC)) mat.LPC[,k] <- -res.LPC[[k]]$a[-1]
	for(k in 1:length(res.LPC)) 
		{
		k.coeff <- -res.LPC[[k]]$k
		mat.LAR[,k] <- log((1-k.coeff)/(1+k.coeff))
		}

	# calculate LPC-based Cepstral coefficients
	mat.lpcep <- lpc2cep(rbind(1, -mat.LPC))
	mat.lpcep <- mat.lpcep[-1,]
	temp <- audio.nor
	temp@left <- x

	# find MFCC using command melfcc from package tuneR
	list.lpmfcc <- melfcc(temp, sr=temp@samp.rate, numcep=temp@samp.rate/1000+2, 
	usecmp=FALSE, modelorder=10, spec_out=TRUE, frames_in_rows=FALSE, preemph=0, 
	wintime=seg.size/temp@samp.rate, hoptime=seg.size*ovlp/temp@samp.rate)
	mat.lpmfcc <- list.lpmfcc[["cepstra"]]

	res.oneFile <- list("LPC"=mat.LPC, "LAR"=mat.LAR, "lpcep"=mat.lpcep, "lpmfcc"=mat.lpmfcc)
	res[[length(res)+1]] <- res.oneFile
	names(res)[length(res)] <- file.name
	print(paste("Finished ", counter, " files out of ", n.files, sep=""), quote=F)
	}

# find mean and variance of feature vectors across time segments
mvecs <- varvecs <- list()
for(speaker in 1:length(res))
	{
	LPC.m <- rowMeans(res[[speaker]][[1]])
	LAR.m <- rowMeans(res[[speaker]][[2]])
	lpcep.m <- rowMeans(res[[speaker]][[3]])
	lpmfcc.m <- rowMeans(res[[speaker]][[4]])
	mvecs[[length(mvecs)+1]] <- list(LPC.m, LAR.m, lpcep.m, lpmfcc.m)
	LPC.v <- apply(res[[speaker]][[1]], 1, var)
	LAR.v <- apply(res[[speaker]][[2]], 1, var)
	lpcep.v <- apply(res[[speaker]][[3]], 1, var)
	lpmfcc.v <- apply(res[[speaker]][[4]], 1, var)
	varvecs[[length(varvecs)+1]] <- list(LPC.v, LAR.v, lpcep.v, lpmfcc.v)
	}

# create matrices for mean vectors and variance vectors (rows represent subject and columns represent features)
means.mat.LPC <- means.mat.LAR <- means.mat.lpcep <- means.mat.lpmfcc <- 
vars.mat.LPC <- vars.mat.LAR <- vars.mat.lpcep <- vars.mat.lpmfcc <- matrix(0, length(mvecs), p)

for(k in 1:length(mvecs))
	{
	means.mat.LPC[k,] <- mvecs[[k]][[1]]
	means.mat.LAR[k,] <- mvecs[[k]][[2]]
	means.mat.lpcep[k,] <- mvecs[[k]][[3]]
	means.mat.lpmfcc[k,] <- mvecs[[k]][[4]]
	vars.mat.LPC[k,] <- varvecs[[k]][[1]]
	vars.mat.LAR[k,] <- varvecs[[k]][[2]]
	vars.mat.lpcep[k,] <- varvecs[[k]][[3]]
	vars.mat.lpmfcc[k,] <- varvecs[[k]][[4]]		
	}

# After providing group labels (Parkinson's Disease patients and healthy controls), 
# the mean or variance feature vectors will be ready for a comparison between 
# groups using the desired machine learning classifier

# Added: export the 80 features in the workbook sheet order.
stopifnot(length(res) > 0)
blocks <- list(
  LPC_means=means.mat.LPC, LAR_means=means.mat.LAR,
  Cep_means=means.mat.lpcep, MFCC_means=means.mat.lpmfcc,
  LPC_vars=vars.mat.LPC, LAR_vars=vars.mat.LAR,
  Cep_vars=vars.mat.lpcep, MFCC_vars=vars.mat.lpmfcc
)
for (nm in names(blocks)) {
  colnames(blocks[[nm]]) <- paste0(nm, ":", sub("_.*", "", nm), 1:10)
}
features <- do.call(cbind, blocks)
stopifnot(ncol(features) == 80, all(is.finite(features)))
ids <- tools::file_path_sans_ext(basename(names(res)))
stopifnot(!anyDuplicated(ids))
dir.create("results/extraction_v1", recursive=TRUE, showWarnings=FALSE)
write.csv(data.frame(sample_id=ids, features, check.names=FALSE),
          "results/extraction_v1/extracted_features.csv", row.names=FALSE)
capture.output(sessionInfo(), file="results/extraction_v1/R_session.txt")
cat("Exported", nrow(features), "recordings and", ncol(features), "features\n")
