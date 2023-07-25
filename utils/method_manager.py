import logging
from methods.er import ER
from methods.mir import MIR
from methods.der import DER
from methods.baseline import BASELINE
from methods.filod import FILOD
from methods.filod_random import FILOD_RANDOM
from methods.filod_random_ema import FILOD_RANDOM_EMA   
from methods.ilod import ILOD
from methods.finetune import FINETUNE
from methods.rodeo import RODEO


logger = logging.getLogger()

def select_method(args, criterion, device, train_transform, test_transform, n_classes, writer):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    
    elif args.mode == "der":
        method = DER(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    
    elif args.mode == "filod":
        method = FILOD(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
        
    elif args.mode == "filod_random":
        method = FILOD_RANDOM(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
        
    elif args.mode == "filod_random_ema":
        method = FILOD_RANDOM_EMA(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    
    elif args.mode == "ilod":
        method = ILOD(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    
    elif args.mode == "finetune":
        method = FINETUNE(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )

    elif args.mode == "rodeo":
        method = RODEO(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )

    # elif args.mode == "clad_der":
    #     method = CLAD_DER(
    #         criterion=None,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         writer=writer,
    #         **kwargs,
    #     )
    # elif args.mode == "clad_filod":
    #     method = CLAD_FILOD(
    #         criterion=None,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         writer=writer,
    #         **kwargs,
    #     )
    elif args.mode == "baseline":
        method = BASELINE(
            criterion=None,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )
    # elif args.mode == "clad_baseline":
    #     method = CLAD_BASELINE(
    #         criterion=None,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         writer=writer,
    #         **kwargs,
    #     )
    # elif args.mode == "clad_filod":
    #     method = CLAD_FILOD(
    #         criterion=None,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         writer=writer,
    #         **kwargs,
    #     )
    # elif args.mode == "clad_baseline":
    #     method = CLAD_BASELINE(
    #         criterion=None,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         writer=writer,
    #         **kwargs,
    #     )
    # elif args.mode == "shift_der":
    #     method=SHIFT_DER(
    #         criterion=None,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         writer=writer,
    #         **kwargs,
    #     )

    # method = CLAD_MIR(
    #     criterion=None,
    #     device=device,
    #     train_transform=train_transform,
    #     test_transform=test_transform,
    #     n_classes=n_classes,
    #     writer=writer,
    #     **kwargs,
    # )
    
    return method
