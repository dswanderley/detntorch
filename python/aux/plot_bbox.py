import matplotlib.pyplot as plt
import matplotlib.patches as patches


rect_mask = fol_mask
rect_mask[slice_y.start:slice_y.stop,
        slice_x.start:slice_x.stop
        ] = 1
Image.fromarray((255*rect_mask).astype(np.uint8)).save("rect_mast.png")


# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(gt_mask)

# Create a Rectangle patch
rect = patches.Rectangle((box[0],box[1]),
                        box[2]-box[0], # width
                        box[3]-box[1], # height
                        linewidth=1,edgecolor='w',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()