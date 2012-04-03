# read_input.py
import sys
import mpi


def write_master(msg, handle=sys.stderr):
    if mpi.rank == 0:
        msg = "[%d] %s" %(mpi.rank, msg)
        handle.write(msg)
        handle.flush()

def write_all(msg, handle=sys.stderr):
    '''Have all processes write msg to sys.stderr; Prepend rank id to the message.'''
    msg = "[%d] %s" %(mpi.rank, msg)
    handle.write(msg)

def open_files(FLAGS):
    ''' Helper function to read_input()
        Try to open files read from arguments for reading '''
    file_handles = { }

    if FLAGS.inverse_dev is not None:
        try:
            file_handles['inverse_dev'] = open(FLAGS.inverse_dev, 'r')
        except:
            write_master("Could not open output file %s for writing\n" %(FLAGS.inverse_dev))
            sys.exit(3)
    if FLAGS.inverse is not None:
        try:
            file_handles['inverse'] = open(FLAGS.inverse, 'r')
        except:
            write_master("Could not open output file %s for writing\n" %(FLAGS.inverse))
            sys.exit(3)

    if FLAGS.out is not None:
        try:
            file_handles['out'] = open(FLAGS.out, 'w')
        except:
            write_master("Could not open output file %s for writing\n" %(FLAGS.out))
            sys.exit(3)
    else:
        file_handles['out'] = sys.stdout

    if FLAGS.a1 is not None:
        try:
            file_handles['a1'] = open(FLAGS.a1, 'r')
        except:
            write_master("Could not open m4-intersection file %s for reading\n" %(FLAGS.a1))
            sys.exit(3)

    if FLAGS.a1_dev is not None:
        try:
            file_handles['a1_dev'] = open(FLAGS.a1_dev, 'r')
        except:
            write_master("Could not open m4-intersection-dev file %s for reading\n" %(FLAGS.a1_dev))
            sys.exit(3)

    if FLAGS.a2 is not None:
        try:
            file_handles['a2'] = open(FLAGS.a2, 'r')
        except:
            write_master("Could not open a2 file %s for reading\n" %(FLAGS.a2))
            sys.exit(3)

    if FLAGS.a2_dev is not None:
        try:
            file_handles['a2_dev'] = open(FLAGS.a2_dev, 'r')
        except:
            write_master("Could not open a2_dev file %s for reading\n" %(FLAGS.a2_dev))
            sys.exit(3)

    if FLAGS.ftrees is not None:
        try:
            file_handles['ftrees'] = open(FLAGS.ftrees, 'r')
        except:
            write_master("Could not open ftrees file %s for reading\n" %(FLAGS.ftrees))
            sys.exit(3)
            
    try:
        file_handles['f'] = open(FLAGS.f, 'r')
    except:
        write_master("Could not open ftrain file %s for reading\n" % FLAGS.f)
        sys.exit(3)

    try:
       file_handles['e'] = open(FLAGS.e, 'r')
    except:
        write_master("Could not open etrain file %s for reading\n" % FLAGS.e)
        sys.exit(3)

    try:
        file_handles['etrees'] = open(FLAGS.etrees, 'r')
    except:
        write_master("Could not open etrees train file %s for reading\n" % FLAGS.etrees)
        sys.exit(3)

    if FLAGS.gold is not None:
        try:
            file_handles['gold'] = open(FLAGS.gold, 'r')
        except:
            write_master("Could not open gold train file %s for reading\n" % FLAGS.gold)
            sys.exit(3)

    if FLAGS.weights is not None:
        try:
            file_handles['weights'] = open(FLAGS.weights, 'r')
        except:
            write_master("Could not open weights file %s for reading\n" % FLAGS.weights)
            sys.exit(3)

    if FLAGS.evcb is not None:
        try:
            file_handles['evcb'] = open(FLAGS.evcb, 'r')
        except:
            write_master("Could not open evcb file %s for reading\n" % FLAGS.evcb)
            sys.exit(3)
    if FLAGS.fvcb is not None:
        try:
            file_handles['fvcb'] = open(FLAGS.fvcb, 'r')
        except:
            write_master("Could not open fvcb file %s for reading\n" %  FLAGS.fvcb)
            sys.exit(3)

    if FLAGS.fdev is not None:
        try:
            file_handles['fdev'] = open(FLAGS.fdev, 'r')
        except:
            sys.stderr.write("Could not open fdev file %s for reading\n" % FLAGS.fdev)
            sys.exit(3)

    if FLAGS.edev is not None:
        try:
            file_handles['edev'] = open(FLAGS.edev, 'r')
        except:
            sys.stderr.write("Could not open edev file %s for reading\n" % FLAGS.edev)
            sys.exit(3)

    if FLAGS.etreesdev is not None:
        try:
            file_handles['etreesdev'] = open(FLAGS.etreesdev, 'r')
        except:
            sys.stderr.write("Could not open etrees dev file %s for reading\n" % FLAGS.etreesdev)
            sys.exit(3)

    if FLAGS.pef is not None:
        try:
            file_handles['pef'] = open(FLAGS.pef, 'r')
        except:
            sys.stderr.write("Could not open pef file %s for reading\n" % FLAGS.pef)
            sys.exit(3)

    if FLAGS.pfe is not None:
        try:
            file_handles['pfe'] = open(FLAGS.pfe, 'r')
        except:
            sys.stderr.write("Could not open pfe file %s for reading\n" % FLAGS.pfe)
            sys.exit(3)

    if FLAGS.ftreesdev is not None:
        try:
            file_handles['ftreesdev'] = open(FLAGS.ftreesdev, 'r')
        except:
            sys.stderr.write("Could not open ftrees dev file %s for reading\n" % FLAGS.ftreesdev)
            sys.exit(3)

    if FLAGS.golddev is not None:
        try:
            file_handles['golddev'] = open(FLAGS.golddev, 'r')
        except:
            sys.stderr.write("Could not open gold dev file %s for reading\n" % FLAGS.golddev)
            sys.exit(3)

    return file_handles
