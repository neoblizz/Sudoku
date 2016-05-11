Sudoku Solver -- GPU Implementation [PPT](https://github.com/neoblizz/Sudoku/blob/master/Sudoku%20Solver%20-%20Presentation.pdf)
===================================
Sudoku is a logic-based number puzzle with the objective of filling an entire 9x9 grid with digits from 1-9 so that each column, row, and 3x3 block has all the numbers once and only once. Since there are 6.67 * 10^21 possible valid solutions for a 9x9 Sudoku puzzle, it becomes an interesting problem to solve on GPUs because of their mass compute ability. The intuitive way to solve sudokus is to use a brute-force search to find solutions. However, any brute-force implementation will be significantly bound by memory constraints on GPUs, and a pure brute-force algorithm would yield very poor performance. To explore the compute ability in solving sudoku problems, it’s interesting to look at other algorithms that efficiently utilize compute ability of GPUs.

While many algorithms exist for human and computer solvers alike, a particularly unique approach to think about when solving Sudoku puzzles on a GPUs is the genetic algorithm model, which is an algorithm based on the theory of evolution. The main idea is to represent a single chromosome as a possible solution for the problem, then randomly swap cells within each block (simulating genetic mutation) to generate several new possible solutions for the problem. A  control unit called the evaluation function then evaluates and assigns ratings to these possible solutions, and the solution with the best ratings is chosen as the parent for the next generation. With continued generation of these solutions and reevaluations, the algorithm eventually converges to an absolute truth. Genetic algorithm is nondeterministic algorithm, it’s an interesting algorithm for solving puzzle problems. Because this algorithm is highly compute intensive, the GPU is a clear choice for development to highlight the efficiency of the genetic algorithm model and to solve interesting puzzles like sudoku!

How to contribute?
==================
- `fork` using GitHub; https://github.com/neoblizz/Sudoku
- Using command line `cd` into a directory you'd like to work on.
- `git clone https://github.com/neoblizz/Sudoku.git`
- `git remote set-url -push origin https://github.com/username/Sudoku.git` This sets the url of the push command to your `username` repository that you forked. That way we can create pull request and make sure nothing accidentally breaks in the main repo. Be sure to change the `username` to your username in the command.
- Make changes to the file you'd like.
- `git add filename`
- `git commit -m "comment here"`
- `git push` You'll be prompted for username and password for your github.
- Once you've pushed the changes on your fork, you can create a pull request on Github to merge the changes.
- Compile `nvcc -std=c++11 -o sudoku sudoku.cu`
- Run `./sudoku 9 -txt Easy_Puzzle.txt`

Milestones
==========
- [x] Project Proposal.

- [x] I/O and Psuedocode.

- [x] Code kernels for implementations discussed.

- [ ] Compare and report results.

- [ ] Develop a real-time graphical interface for algorithm visualization.

Known Issues
==========
- Tree based algorithm is inconclusive as it goes out of memory. Switch techniques from bfs to dfs when tree gets wide.
- Human-logic kernel functions for first few iterations, needs to be debugged.
- Bee Colony determines correct result, needs to be optimized to generate possible solutions per block.

Developers
==========
- Angela Tobin *(University of California, Davis)*
- Chenshan Yuan *(University of California, Davis)*
- Muhammad Osama *(University of California, Davis)*

References
==========
[1] Yuji Sato, Naohiro Hasegawa, and Mikiko Sato. “Acceleration of Genetic Algorithms for Acceleration of Genetic Algorithms for Sudoku Solution on Many-core Processors”<br>
[2] David P. Steensma. “Understanding Chromosome Analysis”<br>
[3] “The Kudoku Sudoku Solver” Web. https://attractivechaos.github.io/plb/kudoku.html<br>
[4] Hilda Huang and Lindsay Zhong. ”Parallel Soduko Solver” Web. http://www.andrew.cmu.edu/user/hmhuang/project_template/finalreport.html<br>
[5] D. Karaboga and B. Akay. Erciyes. “Artificial Bee Colony (ABC), Harmony Search and Bees Algorithms on Numerical Optimization”<br>
[6] QT. Web. http://www.qt.io<br>
[7] GTK. Web. http://www.gtk.org<br>
