# Kompilator Latte

Autor: Andrzej Głuszak (385527)

## Opis
Projekt napisany jest w Ruście. Make powinno działać z zainstalowanym kompilatorem rusta.
Gdyby tak jednak nie było, to `latc` można uruchamiać, pisząc:
```shell
cargo run -- <ścieżka/do/pliku.lat>
```
Biblioteka standardowa latte napisana jest w C. Oprócz wymaganych funkcji zaimplementowałem
`newString` oraz `stringConcat`. Nie można ich wywołać w latte.

Wersja LLVM: 14.

## Biblioteki
- logos - lexer
- lalrpop - generator parserów
- insta - snapshot testing
- ariadne - ładne raportowanie błędów
- anyhow - konwersja błędów
- either, tempfile - używane w testach
- inkwell - rustowe API do LLVM

## Struktura
Projekt podzielony na moduły. 
Przepływ informacji jest standardowy: 
lexer -> parser -> typechecker -> dfa (w tym momencie trochę zbędne, bo robione na poziomie AST)
-> ssa ir (algorytm Brauna) -> codegen llvm

## Rozszerzenia
W tym momencie brak.
