import time
import turtle as t

T=t.Turtle()

t.setup(1000,800,0,0)
T.speed(0)
T.penup()
T.seth(90)
T.fd(340)
T.seth(0)
T.pendown()

T.speed(5)
T.begin_fill()
T.fillcolor('red')
T.circle(50,30)

for i in range(10):
    T.fd(1)
    T.left(10)

T.circle(40,40)

for i in range(6):
    T.fd(1)
    T.left(3)

T.circle(80,40)

for i in range(20):
    T.fd(0.5)
    T.left(5)

T.circle(80,45)

for i in range(10):
    T.fd(2)
    T.left(1)

T.circle(80,25)

for i in range(20):
    T.fd(1)
    T.left(4)

T.circle(50,50)

time.sleep(0.1)

T.circle(120,55)

T.speed(0)

T.seth(-90)
T.fd(70)

T.right(150)
T.fd(20)

T.left(140)
T.circle(140,90)

T.left(30)
T.circle(160,100)

T.left(130)
T.fd(25)

T.penup()
T.right(150)
T.circle(40,80)
T.pendown()

T.left(115)
T.fd(60)

T.penup()
T.left(180)
T.fd(60)
T.pendown()

T.end_fill()

T.right(120)
T.circle(-50,50)
T.circle(-20,90)

T.speed(1)
T.fd(75)

T.speed(0)
T.circle(90,110)

T.penup()
T.left(162)
T.fd(185)
T.left(170)
T.pendown()
T.circle(200,10)
T.circle(100,40)
T.circle(-52,115)
T.left(20)
T.circle(100,20)
T.circle(300,20)
T.speed(1)
T.fd(250)

T.penup()
T.speed(0)
T.left(180)
T.fd(250)
T.circle(-300,7)
T.right(80)
T.circle(200,5)
T.pendown()

T.left(60)
T.begin_fill()
T.fillcolor('green')
T.circle(-80,100)
T.right(90)
T.fd(10)
T.left(20)
T.circle(-63,127)
T.end_fill()

T.penup()
T.left(50)
T.fd(20)
T.left(180)

T.pendown()
T.circle(200,25)

T.penup()
T.right(150)

T.fd(180)

T.right(40)
T.pendown()
T.begin_fill()
T.fillcolor('green')
T.circle(-100,80)
T.right(150)
T.fd(10)
T.left(60)
T.circle(-80,98)
T.end_fill()

T.penup()
T.left(60)
T.fd(13)
T.left(180)

T.pendown()
T.speed(1)
T.circle(-200,23)



t.exitonclick()
