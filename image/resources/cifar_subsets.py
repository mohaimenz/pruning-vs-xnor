#icl stands for incremental learning. All the subsets are taken from cifar-100
def build_subsets(index):
    subset = [];
    multiplier = index // 10;
    for i in range(0, 10):
        for j in range(0, multiplier):
            subset.append(i*10 + j);
    return subset;
    
icl10 = build_subsets(10);
icl20 = build_subsets(20);
icl30 = build_subsets(30);
icl40 = build_subsets(40);
icl50 = build_subsets(50);
icl60 = build_subsets(60);
icl70 = build_subsets(70);
icl80 = build_subsets(80);
icl90 = build_subsets(90);
