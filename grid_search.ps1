$lrs = 0.01, 0.02, 0.03, 0.04
foreach ($lr in $lrs) {
    dvc exp run -S "base.train.lr=$lr"
}