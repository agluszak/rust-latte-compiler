# Kompilator Latte

Autor: Andrzej Głuszak (385527)

## Opis
Projekt napisany jest w Ruście. Make powinno działać z zainstalowanym kompilatorem rusta.
Gdyby tak jednak nie było, to `latc` można uruchamiać, pisząc:
```shell
cargo run -- <ścieżka/do/pliku.lat>
```

## Biblioteki
- chumsky - parser i lexer
- ariadne - ładne wypisywanie błędów


## Struktura
- `src/ast.rs` - AST
- `src/parser.rs` - parser 
- `src/lexer.rs` - lexer
- `src/errors.rs` - obsługa błędów
- `src/typechecker.rs` - typechecker
