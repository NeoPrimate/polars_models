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

---

```py
df.select(
    pl.col("y").stats.glm(
        [
            pl.col("x1"),
            pl.col("x2").pow(2),
            pl.col("x3").fill_null(0),
            pl.col("category").to_dummy(),
        ],
        family: "binomial"
    )
)
```


```py
df.select([
    pl.col("y").stats.ols([pl.col("x1")]).alias("simple_lm"),
    pl.col("y").stats.ols([pl.col("x1"), pl.col("x2"), pl.col("x3")]).alias("complex_lm"),
    pl.col("y").stats.lasso([pl.col("x1"), pl.col("x2")], alpha=0.1).alias("sparse_model")
])
```

```py
df.select(
    pl.col("y").stats.ols(
        [
            pl.col("x1"),
            pl.col("x2").log().alias("log_x2"),
            (pl.col("x3") * pl.col("x4")).alias("interaction_term")
        ]
    )
)
```

```py
df.select(
    pl.col("y").stats.glm(
        [
            pl.col("x1"),
            pl.col("x2").pow(2),
            pl.col("x3").fill_null(0),
            pl.col("category").to_dummy(),
        ],
        family: "binomial"
    ).report().alias("glm_report")
).unnest("glm_report")
```

```py
results = df.select([
    pl.col("y").stats.ols(["x1"]).alias("lm"),
    pl.col("y").stats.lasso(["x1", "x2"]).alias("lasso")
])

lm = results.select(pl.col("lm")).unnest("lm")
lasso = results.select(pl.col("lasso")).unnest("lasso")

comparison = results.select([
    pl.col("lm").struct.field("r_squared").alias("lm_r2"),
    pl.col("lasso").struct.field("r_squared").alias("lasso_r2"),
])
```

### Long Table

```py
df_models = pl.concat([
    df.select(pl.col("y").stats.ols(["x1"]).alias("res")).unnest("res"),
    df.select(pl.col("y").stats.lasso(["x1"]).alias("res")).unnest("res"),
], how="vertical")
```


