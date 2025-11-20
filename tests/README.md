# Suite de Tests para ff5_portfolio

Esta carpeta contiene una suite extensa de tests para el módulo `ff5_portfolio`.

## Estructura de Tests

### Tests por Módulo

- **`test_ff5_portfolio.py`** - Tests básicos para el módulo principal
- **`test_factors_comprehensive.py`** - Tests exhaustivos para `factors.py`
  - Casos normales y límite
  - Manejo de NaN
  - Validaciones de estructura
  - Cálculos de rf_annual
- **`test_optimization_comprehensive.py`** - Tests exhaustivos para `optimization.py`
  - Optimización con diferentes configuraciones
  - Matrices singulares y casos extremos
  - Estabilidad numérica
  - Validación de constraints
- **`test_universe_gio_comprehensive.py`** - Tests exhaustivos para `universe_gio.py`
  - Filtros de sector
  - Filtros de correlación
  - Cálculo de scores
  - Diversificación
- **`test_pipeline_comprehensive.py`** - Tests exhaustivos para `pipeline.py`
  - Integración completa
  - Consistencia entre componentes
  - Validación de outputs
- **`test_data_loader.py`** - Tests básicos para `data_loader.py`
- **`test_data_loader_comprehensive.py`** - Tests exhaustivos para `data_loader.py`
  - Casos de error de red
  - Formatos de datos
  - Normalización de tickers
- **`test_integration.py`** - Tests de integración end-to-end
  - Pipeline completo
  - Consistencia entre módulos
  - Idempotencia

### Configuración

- **`conftest.py`** - Fixtures compartidas para todos los tests

## Ejecutar Tests

### Todos los tests

```bash
pytest tests/ -v
```

### Tests específicos

```bash
# Solo tests de factors
pytest tests/test_factors_comprehensive.py -v

# Solo tests de optimización
pytest tests/test_optimization_comprehensive.py -v

# Solo tests de integración
pytest tests/test_integration.py -v
```

### Con cobertura

```bash
pytest tests/ --cov=ff5_portfolio --cov-report=html
```

### Tests rápidos (sin tests comprehensivos)

```bash
pytest tests/test_ff5_portfolio.py tests/test_data_loader.py -v
```

## Cobertura de Tests

La suite de tests cubre:

- ✅ **Casos normales**: Funcionalidad básica de todos los módulos
- ✅ **Casos límite**: Datos mínimos, máximos, vacíos
- ✅ **Manejo de errores**: Validaciones, excepciones esperadas
- ✅ **Datos faltantes**: NaN, valores faltantes
- ✅ **Estabilidad numérica**: Matrices singulares, valores extremos
- ✅ **Integración**: Pipeline completo end-to-end
- ✅ **Consistencia**: Validación de outputs y relaciones entre componentes

## Estadísticas

- **Total de archivos de test**: 8
- **Total de clases de test**: ~15+
- **Total de tests individuales**: 100+

## Mejores Prácticas

1. **Fixtures compartidas**: Usar fixtures de `conftest.py` cuando sea posible
2. **Semillas aleatorias**: Todos los tests usan `np.random.seed(42)` para reproducibilidad
3. **Nombres descriptivos**: Cada test tiene un docstring explicando qué prueba
4. **Aislamiento**: Cada test es independiente y puede ejecutarse solo
5. **Validaciones múltiples**: Los tests verifican estructura, tipos y valores

## Agregar Nuevos Tests

Al agregar nuevos tests:

1. Seguir la convención de nombres: `test_<modulo>_comprehensive.py`
2. Usar fixtures de `conftest.py` cuando sea posible
3. Incluir docstrings descriptivos
4. Probar casos normales, límite y errores
5. Verificar tipos, estructuras y valores

