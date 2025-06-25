## Sovelluksen asennus ja käynnistäminen
Asenna riippuvuudet komennolla `poetry install` ja tämän jälkeen käynnistä ohjelma repositorion **juuresta** käsin joko komennolla `poetry run python src/main.py` tai komennoilla `poetry shell` ja tämän jälkeen `python src/main.py`. 

## Ohjelman käyttö
Ohjelmassa on komentorivipohjainen käyttöliittymä, joka mahdollistaa erilaisten numeroiden tunnistamiseen tarkoitettujen neuroverkkojen testaamisen. Käyttäjä voi luoda uuden neuroverkon tai kouluttaa aiemmin luotua neuroverkkoa. Tämän lisäksi käyttäjä voi halutessaan listata jo luodut neuroverkot.

## Testien suorittaminen
TODO

## Huomioita
Verkkoa koulutettaessa tulee valita kuinka monta kertaa kaikki koulutusdata syötetään verkolle sekä kuinka paljon jokaisella kierroksella muutetaan verkon parametreja. Melko hyvään tulokseen pitäisi päästä jo ~10 epochilla. Vastaavasti learn_rateksi hyväksi havaitut arvot ovat n. 3 paikkeilla. Liian suuret arvot saattavat johtaa tunnistustarkkuuden heilahteluun ja liian pienet arvot taas saavat verkon oppimaan todella hitaasti.
