from freecell_solver import solve

# tiny solvable setup: just a few cards to prove the algorithm works
cols = [
    [("2","S")],
    [("A","S")],
    [],
    [],
    [],
    [],
    [],
]

path, explored, solved = solve(cols)

if path is None:
    print(f"No solution found (explored {explored} states)")
else:
    print(f"Solved in {len(path)} moves (explored {explored} states):")
    for m in path:
        print(" ", m)
