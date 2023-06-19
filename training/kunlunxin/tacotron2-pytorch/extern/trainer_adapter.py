def create_grad_scaler(args):
    return None

def calculate_loss(model, args, criterion, x, y):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    return loss
