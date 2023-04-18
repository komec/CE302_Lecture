############################################
#
# Statistics for Data Science ############################################
#
# Orneklem  Teorisi ( Sample Theory )
#
# populasyon icinden alt kume secme isidir
#
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

population = np.random.randint(0, 80, 10000)
population[0:10]

# orneklem cekimi
np.random.seed(15)

orneklem = np.random.choice(a =population, size=100)
orneklem[0:10]

orneklem.mean()
population.mean()

orneklem1 = np.random.choice(a = population, size = 100)
orneklem2 = np.random.choice(a = population, size = 100)
orneklem3 = np.random.choice(a = population, size = 100)
orneklem4 = np.random.choice(a = population, size = 100)
orneklem5 = np.random.choice(a = population, size = 100)
orneklem6 = np.random.choice(a = population, size = 100)
orneklem7 = np.random.choice(a = population, size = 100)
orneklem8 = np.random.choice(a = population, size = 100)
orneklem9 = np.random.choice(a = population, size = 100)
orneklem10 = np.random.choice(a = population, size = 100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + \
 orneklem5.mean() + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + \
 orneklem9.mean() + orneklem10.mean() )  / 10

orneklem3.mean()
orneklem8.mean()

##############################################
# BETIMSEL ISTATISTIK

import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

#!pip install researchpy
import researchpy as rp

rp.summary_cont(df[["total_bill","tip","size"]])
rp.summary_cat(df[["sex","smoker","day"]])

# covariance
df[["total_bill","tip"]].cov()

# corrlation
# anlam - siddet -yon
df[["total_bill","tip"]].corr()

# Guven araligi
# ana kutlenin tahmini degerini iceren sayi araligi
#

# PROBABILTY ##########################
# Rassal degisken
# dagilim
# olasilik dagilimi
# olasilik dagilim fonksiyonu

## kesikli olasilik dagilimi
# Bernolli
# Binom
# Poisson

## surekli olasilik dagilimi
# normal
# uniform
# ustel

## BERNOLLI DISTRIBUTION

# f(x:p)
# f olasilik
# 1 -0
# %70 = 0.7

# E(x) = p beklenen deger
# var(x) = pq varyans -  dagilim
from scipy.stats import bernoulli

p = 0.6
aa  = bernoulli(p)
aa.pmf(k = 0)   # probabilty mass function

## BINOMIAL DISTRIBUTION

# f(x : n,p)
from scipy.stats import binom

# reklem harcamasi optimizasyonu
n = 100
p = 0.01

aa = binom(n, p)
print(aa.pmf(1))
print(aa.pmf(5))
print(aa.pmf(10))

## POISSON DISTRIBUTION
# n >> p
# n > 50 ve  nxp < 5   nadir olay

# lamda = beklenen deger = varyans

# # universite not grislerinde hata oluyor
# # 1 yil olcum yapilmis dagilim ve ortalama hata sayisi lambda biliniyor
# # hic hata olmamasi
# # 3 hata ve 5 hata olasiliklari??

from scipy.stats import poisson

lambda_ = 0.1
aa = poisson(mu = lambda_)

print(aa.pmf(k = 0))
print(aa.pmf(k = 3))
print(aa.pmf(k = 5))

## NORMAL Distribution

# mu = ortalama
# alfa = varyans

# ornek
# normal dagilim bilgisi var
# ortalam 80k  std 5k
# 90k ?

from scipy.stats import norm

#90'dan fazla olması
1-norm.cdf(90, 80, 5)

#70'den fazla olması
1-norm.cdf(70, 80, 5)

#73'den az olması
norm.cdf(73, 80, 5)

#85 ile 90 arasında olması
norm.cdf(90, 80, 5) - norm.cdf(85, 80, 5)


#####################################################
#####################################################

# HIPOTEZLER

# H0: μ = 50
# H1: μ ≠ 50
#
# H0: μ <= 50
# H1: μ > 50
#
# H0: μ >= 50
# H1: μ < 50

# pvalue
# p< 0.05

# Hipotez testi adimlari
# 1 adim
# #hipotesleri kuruyorum
# H0: μ = 20  # kisilerin 4 adimda gecirdigi sure 20 dir
# H1: μ ≠ 20  # kisilerin 4 adimda gecirdigi sure 20 degildir

# 2 adim
# anlamlilik duzeyi ve tablo degerinin belirlenmesi
# alfa = 0.05  anlamlilik duzeyi = kabul edilebilir hata miktari

# 3 adim
# test istatistigi
# z hesap degeri bulduk

# 4 adim
# z hesap ile z tablo degerini karsilastiriyorum
# test > tablo ise H0 RED

# 5 adim
# yorum

# ornek
#

# H0: μ = 170
# H1: μ ≠ 170

import numpy as np

olcumler = np.array([17, 160, 234, 149, 145, 107, 197, 75, 201, 225, 211, 119,
              157, 145, 127, 244, 163, 114, 145,  65, 112, 185, 202, 146,
              203, 224, 203, 114, 188, 156, 187, 154, 177, 95, 165, 50, 110,
              216, 138, 151, 166, 135, 155, 84, 251, 173, 131, 207, 121, 120])

olcumler[0:10]

import scipy.stats as stats
stats.describe(olcumler)

import matplotlib
matplotlib.use('TkAgg')

# varsayimlar
# normallik varsayimi

# histogram
import pandas as pd
pd.DataFrame(olcumler).plot.hist()

#qqplot
import pylab
stats.probplot(olcumler, dist="norm", plot=pylab)
pylab.show()

# Shapiro Testi (dagilim uygumlugu testi)
# H0: Örnek dağılımı ile teorik normal dağılım arasında ist. ol. anl. bir fark. yoktur
# H1: ... fark vardır

from scipy.stats import shapiro

shapiro(olcumler)
print("T Hesap İstatistiği: " + str(shapiro(olcumler)[0]))
print("Hesaplanan P-value: " + str(shapiro(olcumler)[1]))

## Hipotez Testinin Uygulanması
# H0: Web sitemizde geçirilen ortalama süre 170'tir
# H1: .. degildir
stats.ttest_1samp(olcumler, popmean = 170)



##############################################################
#################
#  AB testi
# bagimsiz iki orneklem T testi
# iki grup arasinda karsilastirma yapmak icin kullaniyoruz

# ornek
#
# Bağımsız İki Örneklem T Testi hipotezlerimiz
# H0: M1 = M2
# H1: M1 != M2

# #VERI TIPI I
A = pd.DataFrame([30,27,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,
        25,21,18,24,26,33,26,27,28,19,25])
B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,
        32,33,33,33,31,32,33,26,32,33,29])

A_B = pd.concat([A, B], axis = 1)
A_B.columns = ["A","B"]
A_B.head()



#VERI TIPI II
A = pd.DataFrame([30,27,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,
        25,21,18,24,26,33,26,27,28,19,25])
B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,
        32,33,33,33,31,32,33,26,32,33,29])

#A ve A'nın grubu
GRUP_A = np.arange(len(A))
GRUP_A = pd.DataFrame(GRUP_A)
GRUP_A[:] = "A"
A = pd.concat([A, GRUP_A], axis = 1)

#B ve B'nin Grubu
GRUP_B = np.arange(len(B))
GRUP_B = pd.DataFrame(GRUP_B)
GRUP_B[:] = "B"
B = pd.concat([B, GRUP_B], axis = 1)

#Tum veri
AB = pd.concat([A,B])
AB.columns = ["gelir","GRUP"]
print(AB.head())
print(AB.tail())

import seaborn  as sns
sns.boxplot(x="GRUP", y="gelir", data=AB)
plt.show()

# varsayim kontrolu
A_B.head()
AB.head()

# normallik test -  shapiro

shapiro(A_B.A)
# Ho : ornek dagilim ile teorik dagilim arasinda istatistiksel olarak anlamli bir sekilde dagilim farki yoktur
# p > 0.05  Ho reddedilemiyor yani normallik varsayimi saglaniyor o zaman test yapabilirim

shapiro(A_B.B)

#varyans homojenligi varsayımı
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

stats.levene(A_B.A, A_B.B)
# H0 reddedilemez

stats.ttest_ind(A_B["A"], A_B["B"], equal_var = True)
test_istatistigi, pvalue = stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True)
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# yorum = p < 0.05

###########################################
# Varyans Analizi
###########################################
#
# H0: M1 = M2 = M3 (grup ortalamalari arasinda ist anl. farklilik yoktur)
# H1: Fark vardir.
#

A = pd.DataFrame([28,33,30,29,28,29,27,31,30,32,28,33,25,29,27,31,31,30,31,34,30,32,31,34,28,32,31,28,33,29])
B = pd.DataFrame([31,32,30,30,33,32,34,27,36,30,31,30,38,29,30,34,34,31,35,35,33,30,28,29,26,37,31,28,34,33])
C = pd.DataFrame([40,33,38,41,42,43,38,35,39,39,36,34,35,40,38,36,39,36,33,35,38,35,40,40,39,38,38,43,40,42])
dfs = [A, B, C]

ABC = pd.concat(dfs, axis = 1)
ABC.columns = ["GRUP_A","GRUP_B","GRUP_C"]
ABC.head()

from scipy.stats import shapiro
shapiro(ABC["GRUP_A"])
shapiro(ABC["GRUP_B"])
shapiro(ABC["GRUP_C"])

stats.levene(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])

from scipy.stats import f_oneway
f_oneway(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])
print('{:.5f}'.format(f_oneway(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])[1]))

ABC.describe().T