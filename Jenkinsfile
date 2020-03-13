#!groovy

@Library('katsdpjenkins@python2') _
katsdp.killOldJobs()
katsdp.setDependencies([
    'ska-sa/katsdpdockerbase/python2',
    'ska-sa/katpoint/master',
    'ska-sa/katsdptelstate/master'])
katsdp.standardBuild(python3: true, katsdpdockerbase_ref: 'python2')
katsdp.mail('ludwig@ska.ac.za')
