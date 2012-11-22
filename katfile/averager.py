import numpy as np

def block_and_average(vis,weight,flag,avsize):
    
    """ 'block_and_average' does the dirty work of averaging for 'average_visibilities'.
    It blocks an input array in its last axis at a list of array indices calculated
    from the input avsize and the shape of the input visibility array.
    It then weighted-averages the blocked data and returns arrays with
    the last axis rotated to the first axis of the array. For time and channel
    averaging of 2d input arrays (as with 'average_visibilities') the function
    is run twice - to rotate the axes back to their original configuration. Otherwise
    the output can be reshaped after return.

    Inputs
    ------
    vis: array of input visibilities (as defined in 'average_visibilities')
    weight: array of input weights   ( " )
    flag: array of input flags       ( " )
    avsize: int the averaging size along the last axis.

    Outputs
    -------
    av_vis: array of averaged visibilities.
    av_weight: array of averaged weights.
    av_flag: array of averaged_flags.
    indices: array of indices where the blocks were constructed.
    """

    # Get the array indices for blocking (if the avsize is bigger than the
    # array dimension to be averaged, use the size of the array dimension
    # instead).
    indices=range(min(avsize,vis.shape[-1]),vis.shape[-1]+1,min(avsize,vis.shape[-1]))
    
    # Block the data at the given indices along the final axis- omit the final
    # element of the blocked array which is the remainder after equal blocks..
    block_weight = np.array(np.split(weight,indices,axis=-1)[:-1])
    block_vis    = np.array(np.split(vis,indices,axis=-1)[:-1])
    block_flag   = np.array(np.split(flag,indices,axis=-1)[:-1])

    # Workaround for numpy zero weight problem (set blocks with zero weight to weight 1
    # so that an average is returned for these blocks, otherwise numpy returns an error).
    zeroweights = np.where(np.all(block_flag,axis=-1))
    block_weight[zeroweights] = 1.0

    # Average the data
    av_vis = np.average(block_vis,axis=-1,weights=block_weight,returned=False)

    # Undo the weights workaround
    block_weight[zeroweights] = 0.0

    #And get the final weights
    av_weight = np.sum(block_weight,axis=-1)
    
    #Now do the flags
    av_flag = np.all(block_flag,axis=-1)

    return av_vis,av_weight,av_flag,indices


def average_visibilities(vis,weight,flag,timestamps,channel_freqs,timeav=10,chanav=8):

    """ 'average_visibilities' performs the task of averaging of visibilities and
    flags and weights. Visibilities are weight-averaged using the weights in the weight
    array with flagged data set to weight zero. The averaged weights are the sum of the input
    weights for each average block. An average flag is retained if all of the data in an
    averageing block is flagged (the averaged visibility in this case is the unweighted
    average of the input visibilities). In cases where the averaging size in channel or time
    does not evenly divide the size of the input data- the remaining channels or timestamps
    at the end of the array after averaging are discarded. Channels are averaged first and
    the timestamps are second. An array of timestamps and and frequencies corresponding to
    each channel is also directly averaged and returned.


    Inputs
    ------
    vis: array(numtimestamps,numchannels) of complex64.
          The input visibilities to be averaged.
    weight: array(numtimestamps,numchannels) of float32.
          The input weights (used for weighted averaging).
    flag: array(numtimestamps,numchannels) of boolean.
          Input flags (flagged data have weight zero before averaging).
    timestamps: array(numtimestamps) of int.
          The timestamps (in mjd seconds) corresponding to the input data.
    channel_freqs: array(numchannels) of int.
          The frequencies (in Hz) corresponding to the input channels.
    timeav: int.
          The desired averaging size in timestamps.
    chanav: int.
          The desired averaging size in channels.
    
    Outputs
    -------
    av_vis: array(int(numtimestamps/timeav),int(numchannels/chanav)) of complex64.
    av_weight: array(int(numtimestamps/timeav),int(numchannels/chanav)) of float32.
    av_flag: array(int(numtimestamps/timeav),int(numchannels/chanav)) of boolean.
    av_mjd: array(int(numtimestamps/timeav)) of int.
    av_freq: array(int(numchannels)/chanav) of int.

    """

    # Set weight of flagged data to zero
    weight[np.where(flag==True)]=0.0
    
    # Get the channel averaged visibilities, weights and flags
    av_vis_chan,av_weight_chan,av_flag_chan,chan_inds = block_and_average(vis,weight,flag,chanav)
    
    # Do the same on the channel averaged data in time
    av_vis,av_weight,av_flag,time_inds = block_and_average(av_vis_chan,av_weight_chan,av_flag_chan,timeav)

    #print av_vis.shape,av_weight.shape,av_flag.shape
    #get the mjd of the average visibilities
    av_freq = np.average(np.array(np.split(channel_freqs,chan_inds)[:-1]),axis=1)
    av_timestamps = np.average(np.array(np.split(timestamps,time_inds)[:-1]),axis=1)

    return av_vis,av_weight,av_flag,av_timestamps,av_freq
