import os


def read_log(filename):
    speeds, memory = [], 0
    with open(filename, 'r') as fp:
        forwards, backwards, comms, steps = [], [], [], []
        for line in fp.readlines():
            if 'MaxMemAllocated' in line:
                memory = max(memory, float(line.split(',')[-1].split('=')[-1].split('G')[0]))
            elif ' step:' in line:
                _, fw, _, bw, comm, step = line.split('|')
                forwards.append(float(fw.split()[-1]))
                backwards.append(float(bw.split()[-1]))
                comms.append(float(comm.split()[-1]))
                steps.append(float(step.split()[-1]))
                speeds.append(forwards[-1] + backwards[-1] + comms[-1] + steps[-1])
            elif line.startswith('Speed:'):
                speed = float(line.split()[-1].split('m')[0])
                speeds.append(speed)
    em_min = lambda x: min(x) if len(x) > 0 else 0
    speed = em_min(forwards) + em_min(backwards) + em_min(comms) + em_min(steps)
    return speed, em_min(speeds), memory * 1024

if __name__ == '__main__':
    stats = {}
    for filename in os.listdir('log'):
        if not filename.endswith('.log'):
            continue
        model, ps, ndev = filename.split('.')[0].split('_')
        ndev = int(ndev)
        if model not in stats:
            stats[model] = {}
        if ps not in stats[model]:
            stats[model][ps] = {}
        speed1, speed2, memory = read_log(f'log/{filename}')
        stats[model][ps][ndev] = (speed1, speed2, memory)

    for model, model_stats in stats.items():
        for ps, ps_stats in model_stats.items():
            print(f'Model: {model}, PS: {ps}')
            print(f'ndev, speed(ms/iter),  memory(MB)')
            for ndev in sorted(list(ps_stats.keys())):
                print(f'{ndev:3}, {ps_stats[ndev][0]:.3f} ({ps_stats[ndev][1]:.3f}), {ps_stats[ndev][2]:.3f}')
