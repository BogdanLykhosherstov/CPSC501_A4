

import tensorflow as tf
import numpy as np

print("--Get data--")
with np.load("./resources/notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0
 
print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='sigmoid')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=20, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

# save my model
model.save("notMNIST.h5")

# def grabImage():


#     #State of mouse
#     b1 = "up"
#     def b1down(event):
#         global b1
#         b1 = "down"
#     def b1up(event):
#         global b1
#         b1 = "up"
#     def motion(event):
#         if b1 == "down":
#             event.widget.create_oval(event.x,event.y,event.x,event.y, width=16)

   

# #Main to draw window, capture buttons events, and save image
# def main():
#     root = Tk()
#     root.title("Draw")
#     drawing_area = Canvas(root,bg="white",width=28*8,height=28*8)
#     drawing_area.pack()
#     drawing_area.bind("<Motion>", motion)
#     drawing_area.bind("<ButtonPress-1>", b1down)
#     drawing_area.bind("<ButtonRelease-1>", b1up)
#     button=Button(root,fg="green",text="Save",command=lambda:getter(drawing_area))
#     button.pack(side=LEFT)
#     button=Button(root,fg="green",text="Clear",command=lambda:delete(drawing_area))
#     button.pack(side=RIGHT)
#     def delete(widget):
#         widget.delete("all")
#     def getter(widget):
#         x=root.winfo_rootx()+widget.winfo_x()
#         y=root.winfo_rooty()+widget.winfo_y()
#         x1=x+widget.winfo_width()
#         y1=y+widget.winfo_height()
#         grabbed = ImageGrab.grab()
#         grabbed = grabbed.crop((x,y,x1,y1))
#         grabbed = grabbed.resize((28,28))
#         grabbed = grabbed.convert(mode="L")
#         grabbed.save("image.png")
#     root.mainloop()

# if __name__ == "__main__":
#     main()