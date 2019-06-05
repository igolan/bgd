def labels_trick(outputs, labels, criterion):
    """
    Labels trick calculates the loss only on labels which appear on the current mini-batch.
    It is implemented for classification loss types (e.g. CrossEntropyLoss()).
    :param outputs: The DNN outputs of the current mini-batch (torch Tensor).
    :param labels: The ground-truth (correct tags) (torch Tensor).
    :param criterion: Criterion (loss).
    :return: Loss value, after applying the labels trick.
    """
    # Get current batch labels (and sort them for reassignment)
    unq_lbls = labels.unique().sort()[0]
    # Create a copy of the labels to avoid in-place modification
    labels_copy = labels.clone()
    # Assign new labels (0,1 ...) because we will select from the outputs only the columns of labels of the current
    #   mini-batch (outputs[:, unq_lbls]), so their "tagging" will be changed (e.g. column number 3, which corresponds
    #   to label number 3 will become column number 0 if labels 0,1,2 do not appear in the current mini-batch, so its
    #   ground-truth should be changed accordingly to label #0).
    for lbl_idx, lbl in enumerate(unq_lbls):
        labels_copy[labels_copy == lbl] = lbl_idx
    # Calcualte loss only over the heads appear in the batch:
    return criterion(outputs[:, unq_lbls], labels_copy)
