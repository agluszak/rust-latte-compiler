# Kompilator Latte

Autor: Andrzej Głuszak (385527)

## Opis
Projekt napisany jest w Ruście. Make powinno działać z zainstalowanym kompilatorem rusta.
Gdyby tak jednak nie było, to `latc` można uruchamiać, pisząc:
```shell
cargo run -- <ścieżka/do/pliku.lat>
```

## Biblioteki
- logos - lexer
- lalrpop - generator parserów
- insta - snapshot testing
- ariadne - ładne raportowanie błędów

## Struktura
Projekt podzielony na moduły. Przepływ informacji jest standardowy: lexer -> parser -> typechecker -> dfa.
