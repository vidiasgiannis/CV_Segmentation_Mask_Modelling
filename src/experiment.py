import matplotlib.pyplot as plt

def plot_results(images, masks, predictions):
    plt.figure(figsize=(10, 10))
    for i in range(3):
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(masks[i], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(predictions[i], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    plt.show()