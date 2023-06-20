import logging
from methods.clad_er import CLAD_ER
from methods.clad_mir import CLAD_MIR
from methods.clad_der import CLAD_DER
from methods.clad_filod import CLAD_FILOD
from methods.clad_baseline import CLAD_BASELINE
from methods.shift_er import SHIFT_ER

logger = logging.getLogger()

def select_method(args, criterion, device, train_transform, test_transform, n_classes, writer):
    kwargs = vars(args)
    if args.mode == "clad_er":
        method = CLAD_ER(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "clad_mir":
        method = CLAD_MIR(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )

    elif args.mode == "clad_der":
        method = CLAD_DER(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "clad_filod":
        method = CLAD_FILOD(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "clad_baseline":
        method = CLAD_BASELINE(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "shift_er":
        method = SHIFT_ER(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "clad_filod":
        method = CLAD_FILOD(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "clad_baseline":
        method = CLAD_BASELINE(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "shift_er":
            method = SHIFT_ER(
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
