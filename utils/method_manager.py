import logging
# from methods.rainbow_memory import RM
# from methods.ewc import EWCpp
# from methods.mir import MIR
# from methods.clib import CLIB
from methods.clad_er import CLAD_ER

logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes):
    kwargs = vars(args)
    if args.mode == "gdumb":
        from methods.gdumb import GDumb
        method = GDumb(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "rm":
        method = RM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "bic":
        method = BiasCorrection(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "ewc++":
        method = EWCpp(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "clib":
        method = CLIB(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "clad_er":
        method = CLAD_ER(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method
