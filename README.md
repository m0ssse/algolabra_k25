## Sovelluksen asennus ja käynnistäminen
Asenna riippuvuudet komennolla `poetry install` ja tämän jälkeen käynnistä ohjelma repositorion **juuresta** käsin joko komennolla `poetry run python src/main.py` tai aktivoimalla ensin virtuaaliympäristö komennolla `poetry shell` ja tämän jälkeen suorittamalla komento `python src/main.py`. 

## Ohjelman käyttö
Ohjelmassa on komentorivipohjainen käyttöliittymä, joka mahdollistaa erilaisten numeroiden tunnistamiseen tarkoitettujen neuroverkkojen testaamisen. Käyttäjä voi luoda uuden neuroverkon tai kouluttaa aiemmin luotua neuroverkkoa. Tämän lisäksi käyttäjä voi halutessaan listata jo luodut neuroverkot.

## Testien suorittaminen
Testit voi suorittaa (virtuaaliympäristön aktivoinnin jälkeen) komennolla coverage run `--branch -m pytest src`. Tämän jälkeen testikattavuusraportin saa tulostettua komennolla `coverage report -m` tai raportin voi generoida html-muotoon komennolla `coverage html`. Huomaa, että joissain testeissä verkko alustetaan satunnaisilla painoilla, mistä saattaa seurata se, että jotkit testit eivät mene läpi, jos verkon alkuparametrit sattuvat olemaan riittävän lähellä jotain virhefunktion paikallista minimiä.

## Huomioita
Verkkoa koulutettaessa tulee valita kuinka monta kertaa kaikki koulutusdata syötetään verkolle sekä kuinka paljon jokaisella kierroksella muutetaan verkon parametreja. Melko hyvään tulokseen pitäisi päästä jo ~10 epochilla. Vastaavasti learn_rateksi hyväksi havaitut arvot ovat n. 3 paikkeilla. Liian suuret arvot saattavat johtaa tunnistustarkkuuden heilahteluun ja liian pienet arvot taas saavat verkon oppimaan hitaammin.
