// DATA ACC IDEA FILE
Square board[81];

// row is board[i + 9*rowOffset] through board[i+8 + 9*rowOffset]

// col is board[i*9 + colOffset] for (i = 0; i<9; i++)

// block is:  (we are looking at board[n])
	-if (n%9 < 3)
		blockCol = 0; //left side
	-else if (n%9 < 6)
		blockCol = 1; //middle
	-else
		blockCol = 2; //right

	
	-if (n<27)
		blockRow =0; //top
	-else if (n<54)
		blockRow = 1; //middle
	-else
		blockRow = 2; //bottom

	
	//can use this to determine which block we are in, so know
	//exactly which Square to start with

	starter = blockRow*27 + blockCol*3;
	 
	Square block[9] = (board[starter], board[starter+1],
		board[starter+2],board[starter+9], board[starter+10],
		board[starter+11],board[starter+18], board[starter+19],
		board[starter+20]);

