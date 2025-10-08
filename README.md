# Polars Models

### Linear Model (`lm`)

```py
df = pl.DataFrame(...)
model = df.lm(pl.col("y") ~ pl.col("x1") + pl.col("x2"))
results = model.fit()
results.summary()
```

### Generalized Linear Models (`glm`)

```py
df = pl.DataFrame(...)
model = df.glm(pl.col("y") ~ pl.col("x1") + pl.col("x2"), family="binomial")
results = model.fit()
results.summary()
```

### Nonlinear Least Squares (`nls`)

```py
df = pl.DataFrame(...)
model = df.lm(pl.col("y") ~ pl.const("a") * pl.exp(pl.const("b") * pl.col("x1")), start={"a": 1.0,"b": 0.1})
results = model.fit()
results.summary()
```

### Mixed effects / hierarchical models (`lme`, `lmer`)

```py
df = pl.DataFrame(...)
model = df.lm(pl.col("y") ~ pl.col("x1") + pl.col("x2"))
results = model.fit()
results.summary()
```

### PCA (`prcomp`, `princomp`)

```py
df = pl.DataFrame(...)
model = df.princomp([pl.col("x1"), pl.col("x2"), pl.col("x3")])
results = model.fit()
results.summary()
```

### Tree-based & ensemble models (`xgb`)

```py
df = pl.DataFrame(...)
model = df.xgb([pl.col("x1"), pl.col("x2"), pl.col("x3")])
results = model.fit()
results.summary()
```



### Time Series (`ma`, `arma`, `arima`, `varimax`)

```py
df = pl.DataFrame(...)
model = df.arima(pl.col("y"), order=(1, 1, 0))
results = model.fit()
results.summary()
```

```py
df = pl.DataFrame(...)
model = df.varimax(pl.col("y") ~ pl.col("x_1"), order=(1, 1, 0))
results = model.fit()
results.summary()
```





