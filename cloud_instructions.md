Once you are inside the cloud machine, copy-paste this
2.  **Upload Code:**
    ```bash
    scp -P 29054 cortex_cloud_phase2_fix.zip root@82.221.170.234:/root/
    ```

3.  **Connect & Run:**
    ```bash
    ssh root@82.221.170.234 -p 29054
    unzip -o cortex_cloud_phase2_fix.zip
    bash setup_and_run.sh
    ```

## 5. Watch It Fly 🚀
The training will start.
- It will use **all 6 GPUs**.
- It should finish 1000 steps in **~1-1.5 hours**.
- When done, you can download the `logs/` folder using `scp` (reverse of step 3) to see the results.
