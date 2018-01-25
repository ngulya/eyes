import curses

def paint_num(m):
	x = 0
	y = 0
	z = 0
	curses.start_color()
	curses.init_pair(2, curses.COLOR_WHITE,curses.COLOR_GREEN)
	if m < 10:
		l = '| ' + str(m) + ' |'
	elif m < 100:
		l = '| ' + str(m) + '|'
	elif m < 1000:
		l = '|' + str(m) + '|'
	else:
		l = str(z)

	while y < 43:
		x = 0
		while x < 138:
			if z == m:
				myscreen.addstr(y,x,l,curses.color_pair(2))
				return
			x += 6
			z += 1
		y += 2
	

def paint_a():
	x = 0
	y = 0
	z = 0
	curses.start_color()
	curses.init_pair(1, curses.COLOR_WHITE,curses.COLOR_BLACK)
	while y < 43:
		x = 0
		while x < 138:
			if z < 10:
				l = '| ' + str(z) + ' |'
			elif z < 100:
				l = '| ' + str(z) + '|'
			elif z < 1000:
				l = '|' + str(z) + '|'
			else:
				l = str(z)
			myscreen.addstr(y,x,l,curses.color_pair(1))
			x += 6
			z += 1
		y += 2
def MainInKey():
	key = 'X'
	paint_a()
	asS = 1
	myscreen.refresh()  
	while key != ord('q') and key != ord('Q'):
		key = myscreen.getch()
		if key == ord('\n'):
			paint_a()
			paint_num(int(myscreen.getstr(0,0,4)))
			myscreen.refresh()

	curses.endwin() 
	exit()

# #  MAIN LOOP
# try:
# 	myscreen = curses.initscr()
# 	curses.curs_set(0)
# 	curses.cbreak(); 
# 	curses.noecho(); 
# 	MainInKey()
# finally:
# 	curses.endwin()