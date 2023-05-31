import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bbox(image, boxes, labels, label_map=None, save_path=None):
    """
    Visualize bounding boxes on the image.

    Args:
        image (torch.Tensor): The image tensor in shape [C, H, W].
        boxes (torch.Tensor): The bounding boxes tensor in shape [N, 4].
        labels (torch.Tensor): The labels tensor in shape [N].
        label_map (dict, optional): A mapping from label indices to label names. 
            Default is None, in which case labels are converted to string as is.
        save_path (str, optional): The path to save the visualized image. 
            If None, the image will be displayed without saving. Default is None.
    """
    # Convert the image tensor to numpy array
    # Also, convert it from [C, H, W] to [H, W, C] and normalize if necessary
    image = image.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]

    # Create a new figure and a subplot (this is needed to add patches - bounding boxes)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Iterate over all boxes
    for i in range(boxes.shape[0]):
        # Create a rectangle patch
        x1, y1, x2, y2 = boxes[i]
        box = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.5, edgecolor='r', facecolor='none')
        
        # Add the rectangle patch (bounding box) to the subplot
        ax.add_patch(box)

        # Get label
        label = str(labels[i].item())
        if label_map:
            label = label_map.get(label, label)

        # Put text (label)
        plt.text(x1, y1, label, color='white')

    # Save the image or display it
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
