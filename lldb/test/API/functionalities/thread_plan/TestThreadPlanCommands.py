"""
Test that thread plan listing, and deleting works.
"""



import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestThreadPlanCommands(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    @expectedFailureAll(oslist=["linux"])
    def test_thread_plan_actions(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.thread_plan_test()

    def check_list_output(self, command, active_plans = [], completed_plans = [], discarded_plans = []):
        # Check the "thread plan list" output against a list of active & completed and discarded plans.
        # If all three check arrays are empty, that means the command is expected to fail. 

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        num_active = len(active_plans)
        num_completed = len(completed_plans)
        num_discarded = len(discarded_plans)

        interp.HandleCommand(command, result)
        print("Command: %s"%(command))
        print(result.GetOutput())

        if num_active == 0 and num_completed == 0 and num_discarded == 0:
            self.assertFalse(result.Succeeded(), "command: '%s' succeeded when it should have failed: '%s'"%
                             (command, result.GetError()))
            return

        self.assertTrue(result.Succeeded(), "command: '%s' failed: '%s'"%(command, result.GetError()))
        result_arr = result.GetOutput().splitlines()
        num_results = len(result_arr)

        # Match the expected number of elements.
        # Adjust the count for the number of header lines we aren't matching:
        fudge = 0
        
        if num_completed == 0 and num_discarded == 0:
            # The fudge is 3: Thread header, Active Plan header and base plan
            fudge = 3
        elif num_completed == 0 or num_discarded == 0:
            # The fudge is 4: The above plus either the Completed or Discarded Plan header:
            fudge = 4
        else:
            # The fudge is 5 since we have both headers:
            fudge = 5

        self.assertEqual(num_results, num_active + num_completed + num_discarded + fudge,
                             "Too many elements in match arrays for: \n%s\n"%result.GetOutput())
            
        # Now iterate through the results array and pick out the results.
        result_idx = 0
        self.assertIn("thread #", result_arr[result_idx], "Found thread header") ; result_idx += 1
        self.assertIn("Active plan stack", result_arr[result_idx], "Found active header") ; result_idx += 1
        self.assertIn("Element 0: Base thread plan", result_arr[result_idx], "Found base plan") ; result_idx += 1

        for text in active_plans:
            self.assertFalse("Completed plan stack" in result_arr[result_idx], "Found Completed header too early.")
            self.assertIn(text, result_arr[result_idx], "Didn't find active plan: %s"%(text)) ; result_idx += 1

        if len(completed_plans) > 0:
            self.assertIn("Completed plan stack:", result_arr[result_idx], "Found completed plan stack header") ; result_idx += 1
            for text in completed_plans:
                self.assertIn(text, result_arr[result_idx], "Didn't find completed plan: %s"%(text)) ; result_idx += 1

        if len(discarded_plans) > 0:
            self.assertIn("Discarded plan stack:", result_arr[result_idx], "Found discarded plan stack header") ; result_idx += 1
            for text in discarded_plans:
                self.assertIn(text, result_arr[result_idx], "Didn't find completed plan: %s"%(text)) ; result_idx += 1


    def thread_plan_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        # Now set a breakpoint in call_me and step over.  We should have
        # two public thread plans
        call_me_bkpt = target.BreakpointCreateBySourceRegex("Set another here", self.main_source_file)
        self.assertTrue(call_me_bkpt.GetNumLocations() > 0, "Set the breakpoint successfully")
        thread.StepOver()
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, call_me_bkpt)
        self.assertEqual(len(threads), 1, "Hit my breakpoint while stepping over")

        current_id = threads[0].GetIndexID()
        current_tid = threads[0].GetThreadID()
        # Run thread plan list without the -i flag:
        command = "thread plan list %d"%(current_id)
        self.check_list_output (command, ["Stepping over line main.c"], [])

        # Run thread plan list with the -i flag:
        command = "thread plan list -i %d"%(current_id)
        self.check_list_output(command, ["Stepping over line main.c", "Stepping out from"])

        # Run thread plan list providing TID, output should be the same:
        command = "thread plan list -t %d"%(current_tid)
        self.check_list_output(command, ["Stepping over line main.c"])

        # Provide both index & tid, and make sure we only print once:
        command = "thread plan list -t %d %d"%(current_tid, current_id)
        self.check_list_output(command, ["Stepping over line main.c"])

        # Try a fake TID, and make sure that fails:
        fake_tid = 0
        for i in range(100, 10000, 100):
            fake_tid = current_tid + i
            thread = process.GetThreadByID(fake_tid)
            if not thread:
                break
        
        command = "thread plan list -t %d"%(fake_tid)
        self.check_list_output(command)

        # Now continue, and make sure we printed the completed plan:
        process.Continue()
        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonPlanComplete)
        self.assertEqual(len(threads), 1, "One thread completed a step")
        
        # Run thread plan list - there aren't any private plans at this point:
        command = "thread plan list %d"%(current_id)
        self.check_list_output(command, [], ["Stepping over line main.c"])

        # Set another breakpoint that we can run to, to try deleting thread plans.
        second_step_bkpt = target.BreakpointCreateBySourceRegex("Run here to step over again",
                                                                self.main_source_file)
        self.assertTrue(second_step_bkpt.GetNumLocations() > 0, "Set the breakpoint successfully")
        final_bkpt = target.BreakpointCreateBySourceRegex("Make sure we get here on last continue",
                                                          self.main_source_file)
        self.assertTrue(final_bkpt.GetNumLocations() > 0, "Set the breakpoint successfully")

        threads = lldbutil.continue_to_breakpoint(process, second_step_bkpt)
        self.assertEqual(len(threads), 1, "Hit the second step breakpoint")

        threads[0].StepOver()
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, call_me_bkpt)

        result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        interp.HandleCommand("thread plan discard 1", result)
        self.assertTrue(result.Succeeded(), "Deleted the step over plan: %s"%(result.GetOutput()))

        # Make sure the plan gets listed in the discarded plans:
        command = "thread plan list %d"%(current_id)
        self.check_list_output(command, [], [], ["Stepping over line main.c:"])

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, final_bkpt)
        self.assertEqual(len(threads), 1, "Ran to final breakpoint")
        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonPlanComplete)
        self.assertEqual(len(threads), 0, "Did NOT complete the step over plan")

