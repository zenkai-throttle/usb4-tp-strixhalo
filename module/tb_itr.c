// SPDX-License-Identifier: GPL-2.0
/* set the USB4 NHI interrupt throttling rate (REG_INT_THROTTLING_RATE, 256 ns units)
 * on AMD Strix Halo USB4 host routers (1022:158d / 1022:158e). Restores the old values on unload.
 * Test use only. Linux sets 128 us by default (nhi_enable_int_throttling). */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#define ITR_BASE 0x38c00
#define ITR_VECS 16
static int val; module_param(val, int, 0444); MODULE_PARM_DESC(val, "throttle in 256ns units (0 = off)");
static struct { struct pci_dev *pdev; void __iomem *io; u32 old[ITR_VECS]; } devs[4];
static int ndev;
static void apply(struct pci_dev *p)
{
	void __iomem *io; int i;
	if (ndev >= 4) return;
	io = ioremap(pci_resource_start(p, 0) + ITR_BASE, ITR_VECS * 4);
	if (!io) return;
	devs[ndev].pdev = pci_dev_get(p); devs[ndev].io = io;
	for (i = 0; i < ITR_VECS; i++) devs[ndev].old[i] = ioread32(io + 4 * i);
	for (i = 0; i < ITR_VECS; i++) iowrite32(val, io + 4 * i);
	pr_info("tb_itr: %s old=%u new=%u readback=%u\n", pci_name(p), devs[ndev].old[0], val, ioread32(io));
	ndev++;
}
static int __init tb_itr_init(void)
{
	struct pci_dev *p = NULL;
	while ((p = pci_get_device(0x1022, 0x158d, p))) apply(p);
	p = NULL;
	while ((p = pci_get_device(0x1022, 0x158e, p))) apply(p);
	return ndev ? 0 : -ENODEV;
}
static void __exit tb_itr_exit(void)
{
	int d, i;
	for (d = 0; d < ndev; d++) {
		for (i = 0; i < ITR_VECS; i++) iowrite32(devs[d].old[i], devs[d].io + 4 * i);
		pr_info("tb_itr: %s restored %u\n", pci_name(devs[d].pdev), devs[d].old[0]);
		iounmap(devs[d].io); pci_dev_put(devs[d].pdev);
	}
}
module_init(tb_itr_init); module_exit(tb_itr_exit);
MODULE_LICENSE("GPL"); MODULE_DESCRIPTION("USB4 NHI interrupt throttle setter (test)");
