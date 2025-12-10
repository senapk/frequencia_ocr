# Install


## Configuração

```bash
pip install opencv-python numpy opencv-stubs

```

## Uso

```bash
# qtd para quantos elementos mostrar
# preview para o tamanho do preview das células (padrão 30)
# quanto maior a letra do terminal, maior pode ficar o preview, se ficar grande demais, vai quebrar a linha estranho
python process.py --qtd 10 --preview 40
```

## Qual termina usar?

- Se você usar o wezterm, as imagens serão exibidas diretamente no terminal.
- Se você usar outro terminal, as imagens serão salvas na pasta `cells/` para revisão manual.

## Todo

- [x] Carregar a imagem
- [x] Processar a imagem para melhor visibilidade
- [x] Corrigir o alinhamento se foto torta ou paisagem
- [x] Cortar as imagens das entradas individuais (20 matriculas e 20 nomes)
- [x] Detecção e eliminação de bordas
- [ ] Testar o reconhecimento com diferentes tipos de filtros
- [ ] Fazer o código de busca e verificação dos match matrícula nome por proximidade
- [ ] Guardar as imagens dos dados não reconhecidos para revisão manual
- [ ] Criar código para mostrar as imagens interativamente no terminal e permitir reconhecimento manual e inserção da matrícula, seja mostrando por proximidade, quais os prováveis matches ou não.
