def ROC_diagram(counter,model,testX,testy):
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    ns_probs = [0 for _ in range(len(testy))]

    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]

    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

    plt.figure('ROC Diagram',dpi=200)
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.',label=f'{counter} Itreration')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'./results/ROC Diagram{counter}')
    plt.show()

